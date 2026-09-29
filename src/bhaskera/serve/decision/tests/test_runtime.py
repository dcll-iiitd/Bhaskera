import hashlib
import io
import tarfile

import pytest

from bhaskera.serve.decision.runtime import llama_release


def test_runtime_root_follows_env(monkeypatch, tmp_path):
    monkeypatch.setenv(llama_release.CACHE_ENV, str(tmp_path))
    directory = llama_release.install_dir("cuda")
    assert directory.parent == tmp_path
    assert directory.name.startswith("llama-b11081-")


def test_runtime_root_default_is_user_cache(monkeypatch):
    monkeypatch.delenv(llama_release.CACHE_ENV, raising=False)
    assert str(llama_release.runtimes_root()).endswith(".cache/bhaskera/llama.cpp")


def test_pinned_linux_cuda_packages_exist():
    assert llama_release.RELEASE == "b11081"
    assert ("linux", "x64", "cuda") in llama_release.PACKAGES
    assert ("linux", "x64", "cuda12") in llama_release.PACKAGES
    assert llama_release.RUNTIME_DIR_ENV == "BHASKERA_LLAMA_DIR"


def test_pick_rejects_unknown_accelerator():
    with pytest.raises(ValueError):
        llama_release.pick("tpu")


def test_fetch_verifies_sha256(tmp_path):
    source = tmp_path / "src.bin"
    source.write_bytes(b"hello")
    good = hashlib.sha256(b"hello").hexdigest()
    target = llama_release.fetch(source.as_uri(), tmp_path / "out" / "f.bin", good)
    assert target.read_bytes() == b"hello"
    with pytest.raises(ValueError, match="sha256 mismatch"):
        llama_release.fetch(source.as_uri(), tmp_path / "out" / "g.bin", "0" * 64)


def test_unpack_strips_the_top_folder(tmp_path):
    archive = tmp_path / "runtime.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        data = b"lib"
        info = tarfile.TarInfo("llama-b11081/libllama.so")
        info.size = len(data)
        bundle.addfile(info, io.BytesIO(data))
    llama_release.unpack(archive, tmp_path / "dest")
    assert (tmp_path / "dest" / "libllama.so").read_bytes() == b"lib"
