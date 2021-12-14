"""
Benchmark send receive on one machine (UCX < 1.10):
UCX_TLS=tcp,sockcm,cuda_copy,cuda_ipc UCX_SOCKADDR_TLS_PRIORITY=sockcm python \
    send-recv.py --server-dev 2 --client-dev 1 \
    --object_type rmm --reuse-alloc --n-bytes 1GB


Benchmark send receive on one machine (UCX >= 1.10):
UCX_TLS=tcp,cuda_copy,cuda_ipc python send-recv.py \
        --server-dev 2 --client-dev 1 --object_type rmm \
        --reuse-alloc --n-bytes 1GB


Benchmark send receive on two machines (IB testing, UCX < 1.10):
# server process
UCX_NET_DEVICES=mlx5_0:1 UCX_TLS=tcp,sockcm,cuda_copy,rc \
    UCX_SOCKADDR_TLS_PRIORITY=sockcm python send-recv.py \
    --server-dev 0 --client-dev 5 --object_type rmm --reuse-alloc \
    --n-bytes 1GB --server-only --port 13337 --n-iter 100

# client process
UCX_NET_DEVICES=mlx5_2:1 UCX_TLS=tcp,sockcm,cuda_copy,rc \
    UCX_SOCKADDR_TLS_PRIORITY=sockcm python send-recv.py \
    --server-dev 0 --client-dev 5 --object_type rmm --reuse-alloc \
    --n-bytes 1GB --client-only --server-address SERVER_IP --port 13337 \
    --n-iter 100


Benchmark send receive on two machines (IB testing, UCX >= 1.10):
# server process
UCX_MAX_RNDV_RAILS=1 UCX_TLS=tcp,cuda_copy,rc python send-recv.py \
        --server-dev 0 --client-dev 5 --object_type rmm --reuse-alloc \
        --n-bytes 1GB --server-only --port 13337 --n-iter 100

# client process
UCX_MAX_RNDV_RAILS=1 UCX_TLS=tcp,cuda_copy,rc python send-recv.py \
        --server-dev 0 --client-dev 5 --object_type rmm --reuse-alloc \
        --n-bytes 1GB --client-only --server-address SERVER_IP --port 13337 \
        --n-iter 100
"""
import argparse
import asyncio
import cProfile
import multiprocessing as mp
import os
from threading import Lock
from time import perf_counter as clock

from dask.utils import format_bytes, parse_bytes

import ucp
from ucp._libs.arr import Array
from ucp._libs.utils_test import (
    blocking_recv,
    blocking_send,
    non_blocking_recv,
    non_blocking_send,
)

mp = mp.get_context("spawn")


def register_am_allocators(args):
    if not args.enable_am:
        return

    import numpy as np

    ucp.register_am_allocator(lambda n: np.empty(n, dtype=np.uint8), "host")

    if args.object_type == "cupy":
        import cupy as cp

        ucp.register_am_allocator(lambda n: cp.empty(n, dtype=cp.uint8), "cuda")
    elif args.object_type == "rmm":
        import rmm

        ucp.register_am_allocator(lambda n: rmm.DeviceBuffer(size=n), "cuda")


def server(queue, args):
    queue.put(os.getpid())
    # import uvloop
    # uvloop.install()

    if args.server_cpu_affinity >= 0:
        os.sched_setaffinity(0, [args.server_cpu_affinity])

    if args.object_type == "numpy":
        import numpy as xp
    elif args.object_type == "cupy":
        import cupy as xp

        xp.cuda.runtime.setDevice(args.server_dev)
    else:
        import cupy as xp

        import rmm

        rmm.reinitialize(
            pool_allocator=True,
            managed_memory=False,
            initial_pool_size=args.rmm_init_pool_size,
            devices=[args.server_dev],
        )
        xp.cuda.runtime.setDevice(args.server_dev)
        xp.cuda.set_allocator(rmm.rmm_cupy_allocator)

    ucp.init()

    register_am_allocators(args)

    # from concurrent.futures import ThreadPoolExecutor
    # pool = ThreadPoolExecutor(max_workers=2)

    def _thread(ep):
        print(ep)

    def run_pool(corofn, *args):
        loop = asyncio.new_event_loop()
        try:
            print(f"run_pool1: {corofn}, {args}")
            coro = corofn(*args)
            print(f"run_pool2: {coro}")
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    async def run():
        async def server_handler(ep):

            if not args.enable_am:
                msg_recv_list = []
                if not args.reuse_alloc:
                    for _ in range(args.n_iter):
                        msg_recv_list.append(xp.zeros(args.n_bytes, dtype="u1"))
                else:
                    t = Array(xp.zeros(args.n_bytes, dtype="u1"))
                    # t = xp.zeros(args.n_bytes, dtype="u1")
                    for _ in range(args.n_iter):
                        msg_recv_list.append(t)

                assert msg_recv_list[0].nbytes == args.n_bytes

            # t = threading.Thread(target=_thread, args=(ep, ))
            # t.start()
            # t.join()

            finished = [0]
            op_lock = Lock()

            def op_started():
                pass

            def op_completed():
                with op_lock:
                    finished[0] += 1

            def op_completed_with_args(request, exception, finished):
                with op_lock:
                    finished[0] += 1

            # ep._ctx.stop_continuous_ucx_progress()

            tasks = []
            for i in range(args.n_iter):
                # tasks.append(
                #     loop.run_in_executor(pool, run_pool, ep.recv, msg_recv_list[i])
                # )
                # tasks.append(
                #     loop.run_in_executor(pool, run_pool, ep.send, msg_recv_list[i])
                # )
                # await loop.run_in_executor(pool, run_pool, ep.recv, msg_recv_list[i])
                # await loop.run_in_executor(pool, run_pool, ep.send, msg_recv_list[i])
                if args.sync:
                    if args.no_error_handling:
                        blocking_recv(
                            ep._ep.worker,
                            ep._ep,
                            msg_recv_list[i],
                            tag=ep._tags["msg_recv"],
                        )
                        blocking_send(
                            ep._ep.worker,
                            ep._ep,
                            msg_recv_list[i],
                            tag=ep._tags["msg_send"],
                        )
                    else:
                        # ep.sync_recv(msg_recv_list[i])
                        # ep.sync_send(msg_recv_list[i])
                        await ep.sync_recv(msg_recv_list[i])
                        await ep.sync_send(msg_recv_list[i])
                elif args.sync_non_blocking:
                    if args.no_error_handling:
                        non_blocking_recv(
                            ep._ep.worker,
                            ep._ep,
                            msg_recv_list[i],
                            op_started,
                            op_completed,
                            tag=ep._tags["msg_recv"],
                        )
                        non_blocking_send(
                            ep._ep.worker,
                            ep._ep,
                            msg_recv_list[i],
                            op_started,
                            op_completed,
                            tag=ep._tags["msg_send"],
                        )
                    else:
                        await ep.sync_recv(
                            msg_recv_list[i],
                            cb_func=op_completed_with_args,
                            cb_args=(finished,),
                        )
                        await ep.sync_send(
                            msg_recv_list[i],
                            cb_func=op_completed_with_args,
                            cb_args=(finished,),
                        )
                elif args.gather:
                    tasks.append(ep.recv(msg_recv_list[i]))
                    tasks.append(ep.send(msg_recv_list[i]))
                else:
                    if args.enable_am is True:
                        recv = await ep.am_recv()
                        await ep.am_send(recv)
                    else:
                        await ep.recv(msg_recv_list[i])
                        await ep.send(msg_recv_list[i])

            if args.gather:
                await asyncio.wait(tasks)

            if args.sync_non_blocking:
                while finished[0] < args.n_iter * 2:
                    ep._ep.worker.progress()
                    # await asyncio.sleep(0)
                print(f"Finished: {finished[0]}")

                await asyncio.sleep(1)

            # ep._ctx.continuous_ucx_progress()

            await ep.close()
            lf.close()

        lf = ucp.create_listener(server_handler, port=args.port)
        queue.put(lf.port)

        while not lf.closed():
            await asyncio.sleep(0.5)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(run())


def client(queue, port, server_address, args):
    queue.put(os.getpid())
    # import uvloop
    # uvloop.install()
    if args.client_cpu_affinity >= 0:
        os.sched_setaffinity(0, [args.client_cpu_affinity])

    import numpy as np

    if args.object_type == "numpy":
        import numpy as xp
    elif args.object_type == "cupy":
        import cupy as xp

        xp.cuda.runtime.setDevice(args.client_dev)
    else:
        import cupy as xp

        import rmm

        rmm.reinitialize(
            pool_allocator=True,
            managed_memory=False,
            initial_pool_size=args.rmm_init_pool_size,
            devices=[args.client_dev],
        )
        xp.cuda.runtime.setDevice(args.client_dev)
        xp.cuda.set_allocator(rmm.rmm_cupy_allocator)

    ucp.init()

    register_am_allocators(args)

    async def run():
        ep = await ucp.create_endpoint(server_address, port)

        if args.enable_am:
            msg = xp.arange(args.n_bytes, dtype="u1")
        else:
            msg_send_list = []
            msg_recv_list = []
            if not args.reuse_alloc:
                for i in range(args.n_iter):
                    msg_send_list.append(xp.arange(args.n_bytes, dtype="u1"))
                    msg_recv_list.append(xp.zeros(args.n_bytes, dtype="u1"))
            else:
                t1 = Array(xp.arange(args.n_bytes, dtype="u1"))
                t2 = Array(xp.zeros(args.n_bytes, dtype="u1"))
                # t1 = xp.arange(args.n_bytes, dtype="u1")
                # t2 = xp.zeros(args.n_bytes, dtype="u1")
                for i in range(args.n_iter):
                    msg_send_list.append(t1)
                    msg_recv_list.append(t2)
            assert msg_send_list[0].nbytes == args.n_bytes
            assert msg_recv_list[0].nbytes == args.n_bytes

        finished = [0]
        op_lock = Lock()

        def op_started():
            pass

        def op_completed():
            with op_lock:
                finished[0] += 1

        def op_completed_with_args(request, exception, finished):
            with op_lock:
                finished[0] += 1

        # ep._ctx.stop_continuous_ucx_progress()

        if args.cuda_profile:
            xp.cuda.profiler.start()
        times = []

        if args.cprofile is not None:
            pr = cProfile.Profile()
            pr.enable()

        total_time = clock()

        tasks = []
        for i in range(args.n_iter):
            start = clock()

            if args.sync:
                if args.no_error_handling:
                    blocking_send(
                        ep._ep.worker,
                        ep._ep,
                        msg_send_list[i],
                        tag=ep._tags["msg_send"],
                    )
                    blocking_recv(
                        ep._ep.worker,
                        ep._ep,
                        msg_recv_list[i],
                        tag=ep._tags["msg_recv"],
                    )
                else:
                    # ep.sync_send(msg_send_list[i])
                    # ep.sync_recv(msg_recv_list[i])
                    await ep.sync_send(msg_send_list[i])
                    await ep.sync_recv(msg_recv_list[i])
            elif args.sync_non_blocking:
                if args.no_error_handling:
                    non_blocking_send(
                        ep._ep.worker,
                        ep._ep,
                        msg_send_list[i],
                        op_started,
                        op_completed,
                        tag=ep._tags["msg_send"],
                    )
                    non_blocking_recv(
                        ep._ep.worker,
                        ep._ep,
                        msg_recv_list[i],
                        op_started,
                        op_completed,
                        tag=ep._tags["msg_recv"],
                    )
                else:
                    await ep.sync_send(
                        msg_send_list[i],
                        cb_func=op_completed_with_args,
                        cb_args=(finished,),
                    )
                    await ep.sync_recv(
                        msg_recv_list[i],
                        cb_func=op_completed_with_args,
                        cb_args=(finished,),
                    )
            elif args.gather:
                tasks.append(ep.send(msg_send_list[i]))
                tasks.append(ep.recv(msg_recv_list[i]))
            else:
                if args.enable_am:
                    await ep.am_send(msg)
                    await ep.am_recv()
                else:
                    await ep.send(msg_send_list[i])
                    await ep.recv(msg_recv_list[i])

            stop = clock()
            times.append(stop - start)

        if args.gather:
            await asyncio.wait(tasks)

        if args.sync_non_blocking:
            while finished[0] < args.n_iter * 2:
                ep._ep.worker.progress()
                # await asyncio.sleep(0)
            print(f"Finished: {finished[0]}")

        total_time = clock() - total_time

        if args.cprofile is not None:
            pr.disable()
            pr.dump_stats(args.cprofile)

        if args.cuda_profile:
            xp.cuda.profiler.stop()

        # ep._ctx.continuous_ucx_progress()

        if args.sync_non_blocking:
            await asyncio.sleep(1)

        await ep.close()
        queue.put(times)
        queue.put(total_time)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(run())

    times = queue.get()
    total_time = queue.get()
    assert len(times) == args.n_iter
    print("Roundtrip benchmark")
    print("--------------------------")
    print(f"n_iter          | {args.n_iter}")
    print(f"n_bytes         | {format_bytes(args.n_bytes)}")
    print(f"object          | {args.object_type}")
    print(f"reuse alloc     | {args.reuse_alloc}")
    print(f"transfer API    | {'AM' if args.enable_am else 'TAG'}")
    print(f"UCX_TLS         | {ucp.get_config()['TLS']}")
    print(f"UCX_NET_DEVICES | {ucp.get_config()['NET_DEVICES']}")
    print("==========================")
    if args.object_type == "numpy":
        print("Device(s)       | CPU-only")
        s_aff = (
            args.server_cpu_affinity
            if args.server_cpu_affinity >= 0
            else "affinity not set"
        )
        c_aff = (
            args.client_cpu_affinity
            if args.client_cpu_affinity >= 0
            else "affinity not set"
        )
        print(f"Server CPU      | {s_aff}")
        print(f"Client CPU      | {c_aff}")
    else:
        print(f"Device(s)       | {args.server_dev}, {args.client_dev}")
    avg = format_bytes(2 * args.n_iter * args.n_bytes / sum(times))
    med = format_bytes(2 * args.n_bytes / np.median(times))
    total = format_bytes(2 * args.n_iter * args.n_bytes / total_time)
    print(f"Average         | {avg}/s")
    print(f"Median          | {med}/s")
    print(f"Total           | {total}/s ({total_time}s)")
    if not args.no_detailed_report:
        print("--------------------------")
        print("Iterations")
        print("--------------------------")
        for i, t in enumerate(times):
            ts = format_bytes(2 * args.n_bytes / t)
            ts = (" " * (9 - len(ts))) + ts
            print("%03d         |%s/s" % (i, ts))


def parse_args():
    parser = argparse.ArgumentParser(description="Roundtrip benchmark")
    parser.add_argument(
        "-n",
        "--n-bytes",
        metavar="BYTES",
        default="10 Mb",
        type=parse_bytes,
        help="Message size. Default '10 Mb'.",
    )
    parser.add_argument(
        "--n-iter",
        metavar="N",
        default=10,
        type=int,
        help="Number of send / recv iterations (default 10).",
    )
    parser.add_argument(
        "-b",
        "--server-cpu-affinity",
        metavar="N",
        default=-1,
        type=int,
        help="CPU affinity for server process (default -1: not set).",
    )
    parser.add_argument(
        "-c",
        "--client-cpu-affinity",
        metavar="N",
        default=-1,
        type=int,
        help="CPU affinity for client process (default -1: not set).",
    )
    parser.add_argument(
        "-o",
        "--object_type",
        default="numpy",
        choices=["numpy", "cupy", "rmm"],
        help="In-memory array type.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=False,
        action="store_true",
        help="Whether to print timings per iteration.",
    )
    parser.add_argument(
        "-s",
        "--server-address",
        metavar="ip",
        default=ucp.get_address(),
        type=str,
        help="Server address (default `ucp.get_address()`).",
    )
    parser.add_argument(
        "-d",
        "--server-dev",
        metavar="N",
        default=0,
        type=int,
        help="GPU device on server (default 0).",
    )
    parser.add_argument(
        "-e",
        "--client-dev",
        metavar="N",
        default=0,
        type=int,
        help="GPU device on client (default 0).",
    )
    parser.add_argument(
        "--reuse-alloc",
        default=False,
        action="store_true",
        help="Reuse memory allocations between communication.",
    )
    parser.add_argument(
        "--cuda-profile",
        default=False,
        action="store_true",
        help="Setting CUDA profiler.start()/stop() around send/recv "
        "typically used with `nvprof --profile-from-start off "
        "--profile-child-processes`",
    )
    parser.add_argument(
        "--rmm-init-pool-size",
        metavar="BYTES",
        default=None,
        type=int,
        help="Initial RMM pool size (default  1/2 total GPU memory)",
    )
    parser.add_argument(
        "--server-only",
        default=False,
        action="store_true",
        help="Start up only a server process (to be used with --client).",
    )
    parser.add_argument(
        "--client-only",
        default=False,
        action="store_true",
        help="Connect to solitary server process (to be user with --server-only)",
    )
    parser.add_argument(
        "-p",
        "--port",
        default=None,
        help="The port the server will bind to, if not specified, UCX will bind "
        "to a random port. Must be specified when --client-only is used.",
        type=int,
    )
    parser.add_argument(
        "--enable-am",
        default=False,
        action="store_true",
        help="Use Active Message API instead of TAG for transfers",
    )
    parser.add_argument(
        "--no-detailed-report",
        default=False,
        action="store_true",
        help="Disable detailed report per iteration.",
    )
    parser.add_argument(
        "--cprofile",
        default=None,
        help="Name of file to dump cProfile stats. Disabled if no file name is "
        "specified.",
    )
    parser.add_argument(
        "--sync", default=False, action="store_true", help="Transfer in sync mode.",
    )
    parser.add_argument(
        "--sync-non-blocking",
        default=False,
        action="store_true",
        help="Transfer in non-blocking sync mode.",
    )
    parser.add_argument(
        "--no-error-handling",
        default=False,
        action="store_true",
        help="Uses simplified transfer functions not covering endpoint error handling."
        "Applies only to '--sync' and '--sync-non-blocking' modes.",
    )
    parser.add_argument(
        "--gather", default=False, action="store_true", help="Gather async transfers.",
    )

    args = parser.parse_args()
    if args.cuda_profile and args.object_type == "numpy":
        raise RuntimeError(
            "`--cuda-profile` requires `--object_type=cupy` or `--object_type=rmm`"
        )
    return args


def main():
    args = parse_args()
    server_address = args.server_address

    # if you are the server, only start the `server process`
    # if you are the client, only start the `client process`
    # otherwise, start everything

    print(f"Starting benchmark [PID: {os.getpid()}]")

    if args.enable_am and not ucp._libs.ucx_api.is_am_supported():
        print("AM only supported in UCX >= 1.11")
        return

    if not args.client_only:
        # server process
        q1 = mp.Queue()
        p1 = mp.Process(target=server, args=(q1, args))
        p1.start()
        pid = q1.get()
        port = q1.get()
        print(f"Server Running at {server_address}:{port} [PID: {pid}]")
    else:
        port = args.port

    if not args.server_only or args.client_only:
        # client process
        q2 = mp.Queue()
        p2 = mp.Process(target=client, args=(q2, port, server_address, args))
        p2.start()
        pid = q2.get()
        print(f"Client connecting to server at {server_address}:{port} [PID: {pid}]")
        p2.join()
        assert not p2.exitcode

    else:
        p1.join()
        assert not p1.exitcode


if __name__ == "__main__":
    main()
