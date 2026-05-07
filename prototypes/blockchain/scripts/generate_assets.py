"""Utility script to (re)generate devnet artifacts for the PoA test network."""

from __future__ import annotations

import json
from pathlib import Path

from eth_account import Account
from eth_keys import keys

PASSWORD = "password"
BASE_DIR = Path(__file__).resolve().parents[1] / "config"
ACCOUNTS_DIR = BASE_DIR / "accounts"

ACCOUNTS = {
    "sealer": "9ed3a8b1f8e3d457ad2fbc57232ece2b2043614d6f90e1a5f760c1dd1e06d5b3",
    "sender": "1cbb28f6c44b2cf3a3aad0cb1b61bf3eefec9e3ea364ab539cc73f2dc16d8df6",
}

NODE_KEYS = {
    "node1": "c937f492819e93915c93dffa50e1399c049b58289f6a1ebf7af5120a0610b7e6",
    "node2": "22049d4b6b6d911cfef95c0d37db114d001e2a76cc85ee5115e1efc9c35e5ad3",
    "node3": "3f0ee4de9caf35a3317bea82552f2013fbe0904e7242aef8d7311ca4fb22d2e9",
    "node4": "4a75b1219777084ffcf5d94b3c7875024a5bc9c1a40add3521dc1a50637e2b2d",
}


def ensure_dirs() -> None:
    (BASE_DIR / "keystore").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "nodekeys").mkdir(parents=True, exist_ok=True)
    ACCOUNTS_DIR.mkdir(parents=True, exist_ok=True)


def write_accounts() -> dict[str, str]:
    addresses: dict[str, str] = {}
    info: dict[str, dict[str, str]] = {}
    for name, key_hex in ACCOUNTS.items():
        key_bytes = bytes.fromhex(key_hex)
        acct = Account.from_key(key_bytes)
        addresses[name] = acct.address.lower()
        keystore = Account.encrypt(key_bytes, PASSWORD)
        (BASE_DIR / "keystore" / f"{name}.json").write_text(json.dumps(keystore))
        meta = {
            "name": name,
            "address": acct.address.lower(),
            "private_key": f"0x{key_hex}",
        }
        (BASE_DIR / f"{name}_account.json").write_text(json.dumps(meta, indent=2))
        info[name] = meta
        (ACCOUNTS_DIR / f"{name}.key").write_text(key_hex)
        (ACCOUNTS_DIR / f"{name}.address").write_text(acct.address.lower())
    return addresses


def write_nodekeys() -> list[str]:
    enodes: list[str] = []
    for name, key_hex in NODE_KEYS.items():
        (BASE_DIR / "nodekeys" / f"{name}.key").write_text(key_hex)
        priv = keys.PrivateKey(bytes.fromhex(key_hex))
        pub = priv.public_key
        enode = f"enode://{pub.to_hex()[2:]}@{name}:30303?discport=0"
        enodes.append(enode)
    (BASE_DIR / "static-nodes.json").write_text(json.dumps(enodes, indent=2))
    return enodes


def write_genesis(addresses: dict[str, str]) -> None:
    chain_id = 4242
    sealer = addresses["sealer"][2:]
    extra_data = "0x" + ("0" * 64) + sealer + ("0" * 130)
    alloc = {
        addresses["sealer"][2:]: {"balance": hex(10**24)},
        addresses["sender"][2:]: {"balance": hex(10**24)},
    }
    genesis = {
        "config": {
            "chainId": chain_id,
            "homesteadBlock": 0,
            "eip150Block": 0,
            "eip155Block": 0,
            "eip158Block": 0,
            "byzantiumBlock": 0,
            "constantinopleBlock": 0,
            "petersburgBlock": 0,
            "istanbulBlock": 0,
            "muirGlacierBlock": 0,
            "berlinBlock": 0,
            "londonBlock": 0,
            "clique": {"period": 2, "epoch": 30000},
        },
        "difficulty": "0x1",
        "gasLimit": "0x1fffffffffffff",
        "extradata": extra_data,
        "alloc": alloc,
        "coinbase": "0x0000000000000000000000000000000000000000",
        "timestamp": "0x0",
        "parentHash": "0x0000000000000000000000000000000000000000000000000000000000000000",
        "mixHash": "0x0000000000000000000000000000000000000000000000000000000000000000",
        "nonce": "0x0",
    }
    (BASE_DIR / "genesis.json").write_text(json.dumps(genesis, indent=2))
    (BASE_DIR / "metadata.json").write_text(
        json.dumps(
            {
                "chain_id": chain_id,
                "accounts": addresses,
                "nodes": list(NODE_KEYS.keys()),
            },
            indent=2,
        )
    )


def main() -> None:
    ensure_dirs()
    addresses = write_accounts()
    write_nodekeys()
    write_genesis(addresses)
    print("Generated devnet assets:", json.dumps(addresses, indent=2))


if __name__ == "__main__":
    main()

