import io
import logging
import os
import struct
from contextlib import contextmanager

import numpy as np
from tornado.iostream import StreamClosedError
from tornado.tcpclient import TCPClient
from tornado.tcpserver import TCPServer

from distributed.comm.utils import from_frames
from distributed.protocol.utils import pack_frames_prelude, unpack_frames
from distributed.utils import nbytes

import rmm

import ucp

normal_env = {
    "UCX_RNDV_SCHEME": "put_zcopy",
    "UCX_MEMTYPE_CACHE": "n",
    "UCX_TLS": "rc,cuda_copy,cuda_ipc",
    "CUDA_VISIBLE_DEVICES": "0",
}


def set_env():
    os.environ.update(normal_env)


def get_num_gpus():
    import pynvml

    pynvml.nvmlInit()
    ngpus = pynvml.nvmlDeviceGetCount()
    pynvml.nvmlShutdown()
    return ngpus


def get_cuda_devices():
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    else:
        ngpus = get_num_gpus()
        return list(range(ngpus))


@contextmanager
def captured_logger(logger, level=logging.INFO, propagate=None):
    """Capture output from the given Logger.
    """
    if isinstance(logger, str):
        logger = logging.getLogger(logger)
    orig_level = logger.level
    orig_handlers = logger.handlers[:]
    if propagate is not None:
        orig_propagate = logger.propagate
        logger.propagate = propagate
    sio = io.StringIO()
    logger.handlers[:] = [logging.StreamHandler(sio)]
    logger.setLevel(level)
    try:
        yield sio
    finally:
        logger.handlers[:] = orig_handlers
        logger.setLevel(orig_level)
        if propagate is not None:
            logger.propagate = orig_propagate


def cuda_array(size):
    return rmm.DeviceBuffer(size=size)


async def send(ep, frames):
    await ep.send(np.array([len(frames)], dtype=np.uint64))
    await ep.send(
        np.array(
            [hasattr(f, "__cuda_array_interface__") for f in frames], dtype=np.bool
        )
    )
    await ep.send(np.array([nbytes(f) for f in frames], dtype=np.uint64))
    # Send frames
    for frame in frames:
        if nbytes(frame) > 0:
            await ep.send(frame)


async def recv(ep):
    try:
        # Recv meta data
        nframes = np.empty(1, dtype=np.uint64)
        await ep.recv(nframes)
        is_cudas = np.empty(nframes[0], dtype=np.bool)
        await ep.recv(is_cudas)
        sizes = np.empty(nframes[0], dtype=np.uint64)
        await ep.recv(sizes)
    except (ucp.exceptions.UCXCanceled, ucp.exceptions.UCXCloseError) as e:
        msg = "SOMETHING TERRIBLE HAS HAPPENED IN THE TEST"
        raise e(msg)

    # Recv frames
    frames = []
    for is_cuda, size in zip(is_cudas.tolist(), sizes.tolist()):
        if size > 0:
            if is_cuda:
                frame = cuda_array(size)
            else:
                frame = np.empty(size, dtype=np.uint8)
            await ep.recv(frame)
            frames.append(frame)
        else:
            if is_cuda:
                frames.append(cuda_array(size))
            else:
                frames.append(b"")

    msg = await from_frames(frames)
    return frames, msg


async def am_send(ep, frames):
    await ep.am_send(np.array([len(frames)], dtype=np.uint64))
    # Send frames
    for frame in frames:
        await ep.am_send(frame)


async def am_recv(ep):
    try:
        # Recv meta data
        nframes = (await ep.am_recv()).view(np.uint64)
    except (ucp.exceptions.UCXCanceled, ucp.exceptions.UCXCloseError) as e:
        msg = "SOMETHING TERRIBLE HAS HAPPENED IN THE TEST"
        raise e(msg)

    # Recv frames
    frames = []
    for _ in range(nframes[0]):
        frame = await ep.am_recv()
        frames.append(frame)

    msg = await from_frames(frames)
    return frames, msg


class TornadoTCPConnection:
    def __init__(self, stream, client=None):
        self._client = client
        self.stream = stream

    @classmethod
    async def connect(cls, host, port):
        client = TCPClient()
        stream = await client.connect(host, port, max_buffer_size=2 ** 30)
        stream.set_nodelay(True)
        return cls(stream, client=client)

    async def send(self, frames):
        stream = self.stream
        if stream is None:
            raise StreamClosedError()

        frames_nbytes = [nbytes(f) for f in frames]
        frames_nbytes_total = sum(frames_nbytes)

        header = pack_frames_prelude(frames)
        header = struct.pack("Q", nbytes(header) + frames_nbytes_total) + header

        frames = [header, *frames]
        frames_nbytes = [nbytes(header), *frames_nbytes]
        frames_nbytes_total += frames_nbytes[0]

        if frames_nbytes_total < 2 ** 17:
            frames = [b"".join(frames)]
            frames_nbytes = [frames_nbytes_total]

        try:
            for each_frame_nbytes, each_frame in zip(frames_nbytes, frames):
                if each_frame_nbytes:
                    if stream._write_buffer is None:
                        raise StreamClosedError()

                    if isinstance(each_frame, memoryview):
                        each_frame = memoryview(each_frame).cast("B")

                    stream._write_buffer.append(each_frame)
                    stream._total_write_index += each_frame_nbytes

            stream.write(b"")
        except StreamClosedError:
            self.stream = None
            self._closed = True
        except Exception() as e:
            raise e

        return frames_nbytes_total

    async def recv(self):
        stream = self.stream
        if stream is None:
            raise Exception("Connection closed")

        fmt = "Q"
        fmt_size = struct.calcsize(fmt)

        try:
            frames_nbytes = await stream.read_bytes(fmt_size)
            (frames_nbytes,) = struct.unpack(fmt, frames_nbytes)

            frames = bytearray(frames_nbytes)
            n = await stream.read_into(frames)
            assert n == frames_nbytes, (n, frames_nbytes)
        except StreamClosedError:
            self.stream = None
            self._closed = True
        except Exception as e:
            raise e
        else:
            try:
                frames = unpack_frames(frames)

                msg = await from_frames(
                    frames,
                    deserializers=("cuda", "dask", "pickle", "error"),
                    allow_offload=True,
                )
            except EOFError:
                raise Exception("aborted stream on truncated data")
            return msg

    async def close(self):
        self.stream.close()
        self._closed = True

    def closed(self):
        return self._closed


class TornadoTCPServer:
    def __init__(self, server, connections, port):
        server.handle_stream = self._handle_stream
        self.server = server
        self._connections = connections
        self._port = port

    async def _handle_stream(self, stream, address):
        self._connections.append(TornadoTCPConnection(stream))

    @classmethod
    async def start_server(cls, host, port):
        connections = []

        server = TCPServer(max_buffer_size=2 ** 30)

        if port is None:

            def _try_listen(server, host):
                while True:
                    try:
                        import random

                        port = random.randint(10000, 60000)
                        server.listen(port, host)
                        return port
                    except OSError:
                        pass

            port = _try_listen(server, host)
        else:
            server.listen(port, host)

        server.start()

        return cls(server, connections, port)

    def get_connections(self):
        return self._connections

    @property
    def port(self):
        return self._port

    def close(self):
        self.server.stop()
        self._closed = True

    def closed(self):
        return self._closed
