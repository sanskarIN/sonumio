"""Selection never extracts an archive or publishes unverified PCM."""
import hashlib
from dataclasses import replace
import json
import numpy as np
import pytest

from sonumio.audio_core.asset_pack import AssetRecipe,SoundPackPlan
from sonumio.audio_core.asset_archive import create_pack_bytes
from sonumio.audio_core.asset_selection import select_asset,list_pack_assets,export_pack_asset
from sonumio.audio_core.asset_pack_cli import main
from sonumio.audio_core.errors import AudioContractError,AudioIntegrityError
from sonumio.audio_core.waveio import inspect_wav_bytes


@pytest.fixture
def library(tmp_path):
    plan=SoundPackPlan("demo",(
        AssetRecipe("rain",4800,variants=2,tags=("ambient","rain")),
        AssetRecipe("wind",4800,nature_kind="wind",tags=("ambient","wind")),
    ))
    raw,receipt=create_pack_bytes(plan)
    path=tmp_path/"demo.zip"
    path.write_bytes(raw)
    return path,raw,receipt


def test_exact_selection(library):
    _,raw,receipt=library
    selected=select_asset(raw,"rain",2,expected_archive_sha256=receipt.archive_sha256)
    assert selected.record["member"]=="audio/rain-002.wav"
    assert selected.record["wav_sha256"]==hashlib.sha256(selected.wav).hexdigest()
    assert inspect_wav_bytes(selected.wav)[0].frames==4800
    assert not selected.as_dict()["archive_extracted"]
    assert selected.wav!=select_asset(raw,"rain",1).wav


@pytest.mark.parametrize("asset,variant",[("../rain",1),("missing",1),("rain",0),("rain",3),("rain",True)])
def test_invalid_selector(library,asset,variant):
    with pytest.raises(AudioContractError):
        select_asset(library[1],asset,variant)


def test_filter_verified_tags_and_sources(library):
    raw=library[1]
    assert list_pack_assets(raw,tags=("ambient",))["matched_assets"]==3
    report=list_pack_assets(raw,tags=("ambient","rain"))
    assert report["matched_assets"]==2
    assert all(a["asset_id"]=="rain" for a in report["assets"])
    assert list_pack_assets(raw,source="procedural")["matched_assets"]==0
    with pytest.raises(AudioContractError):
        list_pack_assets(raw,tags=("rain","rain"))
    with pytest.raises(AudioContractError):
        list_pack_assets(raw,source="unknown")


def test_pinned_export_and_no_clobber(library,tmp_path):
    path,raw,receipt=library
    selected=select_asset(raw,"rain",2)
    output=tmp_path/"chosen.wav"
    report=export_pack_asset(path,output,"rain",2,
        expected_archive_sha256=receipt.archive_sha256,
        expected_wav_sha256=selected.record["wav_sha256"])
    assert report["output_sha256"]==selected.record["wav_sha256"]
    assert output.read_bytes()==selected.wav
    with pytest.raises(AudioContractError):
        export_pack_asset(path,output,"wind")
    with pytest.raises(AudioIntegrityError):
        export_pack_asset(path,output,"wind",overwrite=True,expected_wav_sha256="0"*64)
    assert output.read_bytes()==selected.wav
    assert path.read_bytes()==raw


def test_bad_archive_or_pin_never_creates_parent(library,tmp_path):
    path,raw,_=library
    output=tmp_path/"missing"/"chosen.wav"
    with pytest.raises(AudioIntegrityError):
        export_pack_asset(path,output,"rain",expected_archive_sha256="0"*64,create_parent=True)
    assert not output.parent.exists()
    path.write_bytes(raw+b"trailing")
    with pytest.raises(AudioIntegrityError):
        export_pack_asset(path,output,"rain",create_parent=True)
    assert not output.parent.exists()


def test_list_and_export_cli(library,tmp_path,capsys):
    path,_,receipt=library
    assert main(["sound-pack-list",str(path),"--tag","rain"])==0
    assert json.loads(capsys.readouterr().out)["matched_assets"]==2
    target=tmp_path/"selected.wav"
    assert main(["sound-pack-select",str(path),"--asset","rain","--variant","2",
        "--output",str(target),"--expected-sha256",receipt.archive_sha256])==0
    report=json.loads(capsys.readouterr().out)
    assert target.exists() and report["asset"]["variant"]==2
    assert main(["sound-pack-select",str(path),"--asset","missing","--output",str(tmp_path/"bad.wav")])==2
    assert "not present" in capsys.readouterr().err
