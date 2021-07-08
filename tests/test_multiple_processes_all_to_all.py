import asyncio
import multiprocessing
from time import monotonic

import numpy as np
import pytest
from tornado import gen
from tornado.ioloop import IOLoop
from utils import (
    AsyncioCommConnection,
    AsyncioCommServer,
    TornadoTCPConnection,
    TornadoTCPServer,
    UCXConnection,
    UCXServer,
)

from dask.utils import format_bytes
from distributed.utils import nbytes

import ucp

GatherSendRecv = False
Iterations = 3
Size = 2 ** 20

OP_NONE = 0
OP_WORKER_LISTENING = 1
OP_CLUSTER_READY = 2
OP_WORKER_COMPLETED = 3
OP_SHUTDOWN = 4


class UCXProcess:
    def __init__(
        self,
        signal,
        ports,
        lock,
        worker_num,
        num_workers,
        endpoints_per_worker,
        monitor_port,
        transfer_to_cache,
    ):
        self.signal = signal
        self.ports = ports
        self.lock = lock
        self.worker_num = worker_num
        self.num_workers = num_workers
        self.endpoints_per_worker = endpoints_per_worker

        # UCX only effectively creates endpoints at first transfer, but this
        # isn't necessary for tornado/asyncio.
        self.transfer_to_cache = transfer_to_cache

        self.monitor_port = monitor_port

        self.conns = dict()
        self.connections = set()
        self.bytes_bandwidth = dict()

        self.cluster_started = False

    async def _sleep(self, delay):
        await asyncio.sleep(delay)

    async def _gather(self, tasks):
        await asyncio.gather(*tasks)

    async def _transfer(self, ep, msg2send, send_first=True):
        for i in range(Iterations):
            if GatherSendRecv:
                msgs = [ep.recv(), ep.send(msg2send)]
                await self._gather(msgs)
            else:
                # This seems to be faster!
                if send_first:
                    await ep.send(msg2send)
                    await ep.recv()
                else:
                    await ep.recv()
                    await ep.send(msg2send)

    async def _listener(self, ep):
        message = np.arange(Size, dtype=np.uint8)

        await self._transfer(ep, message)

    async def _client(self, my_port, worker_address, ep, cache_only=False):
        message = np.arange(Size, dtype=np.uint8)
        send_recv_bytes = (nbytes(message) * 2) * Iterations

        t = monotonic()
        await self._transfer(ep, message, send_first=False)
        total_time = monotonic() - t

        if cache_only is False:
            self.bytes_bandwidth[worker_address].append(
                (send_recv_bytes, send_recv_bytes / total_time)
            )

    def _init(self):
        ucp.init()

    async def _create_listener(self, host, port=None, cb=None):
        return await UCXServer.start_server(host, port, cb or self._listener)

    async def _monitor_listener_cb(self, ep):
        pass

    async def _create_endpoint(self, host, port):
        my_port = self.listener.port
        my_task = "Worker"
        remote_task = "Worker"

        while True:
            try:
                ep = await UCXConnection.open_connection(host, port)
                return ep
            except ucp.exceptions.UCXCanceled as e:
                print(
                    "%s[%d]->%s[%d] Failed: %s"
                    % (my_task, my_port, remote_task, port, e),
                    flush=True,
                )
                await self._sleep(0.1)

    def get_connections(self):
        return self.listener._connections

    async def run_monitor(self):
        self._init()

        self.listener_address = ucp.get_address(ifname="enp1s0f0")
        self.listener = await self._create_listener(
            self.listener_address, self.monitor_port, self._monitor_listener_cb,
        )

        with self.lock:
            self.signal[0] = self.listener.port

        # Wait for all workers to connect
        while len(self.get_connections()) != (
            self.endpoints_per_worker * self.num_workers
        ):
            await self._sleep(0.1)

        # Get all worker addresses
        worker_addresses = []
        for conn in self.get_connections():
            _, address = await conn.recv()
            address = address["data"]
            assert address[0] == OP_WORKER_LISTENING
            worker_addresses.append((address[1], address[2]))

        # Send a list of all worker addresses to each worker, indicating the cluster
        # is ready
        for conn in self.get_connections():
            await conn.send([OP_CLUSTER_READY, worker_addresses])

        # Wait for all workers to complete
        for conn in self.get_connections():
            _, complete = await conn.recv()
            complete = complete["data"]
            assert int(complete[0]) == OP_WORKER_COMPLETED

        # Signal all workers to shutdown
        for conn in self.get_connections():
            await conn.send((OP_SHUTDOWN,))

        for conn in self.get_connections():
            await conn.close()

        self.listener.close()

        # Wait for a shutdown signal from monitor
        try:
            while not self.listener.closed():
                await self._sleep(0.1)
        except ucp.UCXCloseError:
            pass

    async def _wait_for_workers(self):
        if self.monitor_port == 0:
            with self.lock:
                self.signal[0] += 1
                self.ports[self.worker_num] = self.listener.port

            while self.signal[0] != self.num_workers:
                await self._sleep(0.1)
        else:
            self.monitor_ep = await self._create_endpoint(
                self.listener_address, self.monitor_port
            )
            await self.monitor_ep.send(
                (OP_WORKER_LISTENING, self.listener_address, self.listener.port)
            )
            _, worker_addresses = await self.monitor_ep.recv()
            assert worker_addresses["data"][0] == OP_CLUSTER_READY
            self.worker_addresses = worker_addresses["data"][1]
            assert len(self.worker_addresses) == self.num_workers

    async def _wait_for_completion(self):
        if self.monitor_port == 0:
            with self.lock:
                self.signal[1] += 1
                self.ports[self.worker_num] = self.listener.port

            while self.signal[1] != self.num_workers:
                await self._sleep(0)
        else:
            await self.monitor_ep.send((OP_WORKER_COMPLETED,))
            _, shutdown = await self.monitor_ep.recv()
            shutdown = shutdown["data"]
            assert shutdown[0] == OP_SHUTDOWN

    async def _create_all_endpoints(self):
        for i in range(self.endpoints_per_worker):
            client_tasks = []
            # Create endpoints to all other workers
            if self.monitor_port == 0:
                for remote_port in list(self.ports):
                    if remote_port == self.listener.port:
                        continue

                    ep = await self._create_endpoint(self.listener_address, remote_port)
                    self.bytes_bandwidth[remote_port] = []
                    self.conns[(remote_port, i)] = ep

                    if self.transfer_to_cache:
                        client_tasks.append(
                            self._client(
                                self.listener.port, remote_port, ep, cache_only=True
                            )
                        )
            else:
                for worker_address in self.worker_addresses:
                    if (
                        worker_address[0] == self.listener_address
                        and worker_address[1] == self.listener.port
                    ):
                        continue

                    ep = await self._create_endpoint(*worker_address)
                    self.bytes_bandwidth[worker_address] = []
                    self.conns[(worker_address, i)] = ep

                    if self.transfer_to_cache:
                        client_tasks.append(
                            self._client(
                                self.listener.port, worker_address, ep, cache_only=True
                            )
                        )

            if self.transfer_to_cache:
                await self._gather(client_tasks)

    async def _wait_for_connections_cache(self):
        # Wait until listener->ep connections have all been cached
        while len(self.get_connections()) != self.endpoints_per_worker * (
            self.num_workers - 1
        ):
            await self._sleep(0.1)

    async def _exchange_messages(self):
        # Exchange messages with other workers
        client_tasks = []
        listener_tasks = []
        for (worker_address, _), ep in self.conns.items():
            client_tasks.append(self._client(self.listener.port, worker_address, ep))
        for listener_ep in self.get_connections():
            listener_tasks.append(self._listener(listener_ep))
        all_tasks = client_tasks + listener_tasks
        await self._gather(all_tasks)

    async def _close_connections_and_listener(self):
        for conn in self.get_connections():
            await conn.close()

        self.listener.close()

        # Wait for a shutdown signal from monitor
        try:
            while not self.listener.closed():
                await self._sleep(0.1)
        except ucp.UCXCloseError:
            pass

    async def run(self):
        self._init()

        # Start listener
        self.listener_address = ucp.get_address(ifname="enp1s0f0")
        self.listener = await self._create_listener(self.listener_address)

        await self._wait_for_workers()

        await self._create_all_endpoints()

        await self._wait_for_connections_cache()

        await self._exchange_messages()

        await self._wait_for_completion()

        await self._close_connections_and_listener()

    def get_results(self):
        for remote_port, bb in self.bytes_bandwidth.items():
            total_bytes = sum(b[0] for b in bb)
            avg_bandwidth = np.mean(list(b[1] for b in bb))
            median_bandwidth = np.median(list(b[1] for b in bb))
            print(
                "[%d, %s] Transferred bytes: %s, average bandwidth: %s/s, "
                "median bandwidth: %s/s"
                % (
                    self.listener.port,
                    remote_port,
                    format_bytes(total_bytes),
                    format_bytes(avg_bandwidth),
                    format_bytes(median_bandwidth),
                )
            )


class TornadoProcess(UCXProcess):
    def __init__(
        self,
        signal,
        ports,
        lock,
        worker_num,
        num_workers,
        endpoints_per_worker,
        monitor_port,
    ):
        super().__init__(
            signal,
            ports,
            lock,
            worker_num,
            num_workers,
            endpoints_per_worker,
            monitor_port,
            transfer_to_cache=False,
        )

    async def _sleep(self, delay):
        await gen.sleep(delay)

    async def _gather(self, tasks):
        await gen.multi(tasks)

    def _init(self):
        return

    async def _create_listener(self, host, port=None, cb=None):
        return await TornadoTCPServer.start_server(host, port=port)

    async def _create_endpoint(self, host, port):
        return await TornadoTCPConnection.connect(host, port)


class AsyncioProcess(UCXProcess):
    def __init__(
        self,
        signal,
        ports,
        lock,
        worker_num,
        num_workers,
        endpoints_per_worker,
        monitor_port,
    ):
        super().__init__(
            signal,
            ports,
            lock,
            worker_num,
            num_workers,
            endpoints_per_worker,
            monitor_port,
            transfer_to_cache=False,
        )

    def _init(self):
        return

    async def _create_listener(self, host, port=None, cb=None):
        return await AsyncioCommServer.start_server(host, port)

    async def _create_endpoint(self, host, port):
        host = ucp.get_address(ifname="enp1s0f0")
        return await AsyncioCommConnection.open_connection(host, port)


def ucx_process(
    signal,
    ports,
    lock,
    worker_num,
    num_workers,
    endpoints_per_worker,
    is_monitor,
    monitor_port,
):
    w = UCXProcess(
        signal,
        ports,
        lock,
        worker_num,
        num_workers,
        endpoints_per_worker,
        monitor_port,
        transfer_to_cache=True,
    )
    run_func = w.run_monitor if is_monitor else w.run
    asyncio.get_event_loop().run_until_complete(run_func())
    w.get_results()


def asyncio_process(
    signal,
    ports,
    lock,
    worker_num,
    num_workers,
    endpoints_per_worker,
    is_monitor,
    monitor_port,
):
    w = AsyncioProcess(
        signal,
        ports,
        lock,
        worker_num,
        num_workers,
        endpoints_per_worker,
        monitor_port,
    )
    run_func = w.run_monitor if is_monitor else w.run
    asyncio.get_event_loop().run_until_complete(run_func())
    w.get_results()


def tornado_process(
    signal,
    ports,
    lock,
    worker_num,
    num_workers,
    endpoints_per_worker,
    is_monitor,
    monitor_port,
):
    w = TornadoProcess(
        signal,
        ports,
        lock,
        worker_num,
        num_workers,
        endpoints_per_worker,
        monitor_port,
    )
    run_func = w.run_monitor if is_monitor else w.run
    IOLoop.current().run_sync(run_func)
    w.get_results()


def _test_send_recv_cu(
    num_workers, endpoints_per_worker, enable_monitor, communication
):
    ctx = multiprocessing.get_context("spawn")

    monitor_port = 0

    signal = ctx.Array("i", [0, 0])
    ports = ctx.Array("i", range(num_workers))
    lock = ctx.Lock()

    if enable_monitor:
        monitor_process = ctx.Process(
            name="worker",
            target=communication,
            args=[signal, ports, lock, 0, num_workers, endpoints_per_worker, True, 0],
        )
        monitor_process.start()

        while signal[0] == 0:
            pass

        monitor_port = signal[0]

    worker_processes = []
    for worker_num in range(num_workers):
        worker_process = ctx.Process(
            name="worker",
            target=communication,
            args=[
                signal,
                ports,
                lock,
                worker_num,
                num_workers,
                endpoints_per_worker,
                False,
                monitor_port,
            ],
        )
        worker_process.start()
        worker_processes.append(worker_process)

    for worker_process in worker_processes:
        worker_process.join()

    if enable_monitor:
        monitor_process.join()

    assert worker_process.exitcode == 0


@pytest.mark.parametrize("num_workers", [2, 4, 8])
@pytest.mark.parametrize("endpoints_per_worker", [1])
@pytest.mark.parametrize("enable_monitor", [True, False])
@pytest.mark.parametrize(
    "communication", [ucx_process, asyncio_process, tornado_process]
)
def test_send_recv_cu(num_workers, endpoints_per_worker, enable_monitor, communication):
    _test_send_recv_cu(num_workers, endpoints_per_worker, enable_monitor, communication)
