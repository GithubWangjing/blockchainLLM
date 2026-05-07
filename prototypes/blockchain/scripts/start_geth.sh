#!/bin/sh
set -euo pipefail

DATADIR=${DATADIR:-/data}
CONFIG_DIR=${CONFIG_DIR:-/config}
CHAIN_ID=${CHAIN_ID:-4242}
P2P_PORT=${P2P_PORT:-30303}
HTTP_ENABLED=${HTTP_ENABLED:-true}
WS_ENABLED=${WS_ENABLED:-true}
HTTP_PORT=${HTTP_PORT:-8545}
WS_PORT=${WS_PORT:-8546}
IMPORT_ACCOUNT=${IMPORT_ACCOUNT:-false}
ENABLE_MINER=${ENABLE_MINER:-false}
ACCOUNT_PASSWORD_FILE=${ACCOUNT_PASSWORD_FILE:-$CONFIG_DIR/password.txt}
ACCOUNT_KEY_FILE=${ACCOUNT_KEY_FILE:-$CONFIG_DIR/accounts/sealer.key}
ACCOUNT_ADDRESS_FILE=${ACCOUNT_ADDRESS_FILE:-$CONFIG_DIR/accounts/sealer.address}

if [ ! -d "$DATADIR/geth" ]; then
  geth --datadir "$DATADIR" init "$CONFIG_DIR/genesis.json"
fi

if [ -f "$CONFIG_DIR/static-nodes.json" ]; then
  cp "$CONFIG_DIR/static-nodes.json" "$DATADIR/static-nodes.json"
fi

if [ "$IMPORT_ACCOUNT" = "true" ] && [ -f "$ACCOUNT_KEY_FILE" ]; then
  if ! ls "$DATADIR"/keystore/UTC* >/dev/null 2>&1; then
    geth --datadir "$DATADIR" account import \
      --password "$ACCOUNT_PASSWORD_FILE" \
      "$ACCOUNT_KEY_FILE" >/dev/null
  fi
fi

HTTP_OPTS=""
if [ "$HTTP_ENABLED" = "true" ]; then
  HTTP_OPTS="--http --http.addr 0.0.0.0 --http.port $HTTP_PORT --http.api eth,net,web3,txpool,clique --http.vhosts=* --http.corsdomain=*"
fi

WS_OPTS=""
if [ "$WS_ENABLED" = "true" ]; then
  WS_OPTS="--ws --ws.addr 0.0.0.0 --ws.port $WS_PORT --ws.api eth,net,web3 --ws.origins=*"
fi

MINER_OPTS=""
if [ "$ENABLE_MINER" = "true" ] && [ -f "$ACCOUNT_ADDRESS_FILE" ]; then
  SEALER_ADDRESS=$(cat "$ACCOUNT_ADDRESS_FILE")
  MINER_OPTS="--mine --miner.threads=1 --unlock $SEALER_ADDRESS --allow-insecure-unlock --password $ACCOUNT_PASSWORD_FILE --miner.gaslimit 12000000"
fi

exec geth \
  --datadir "$DATADIR" \
  --networkid "$CHAIN_ID" \
  --syncmode full \
  --gcmode full \
  --port "$P2P_PORT" \
  --authrpc.addr 0.0.0.0 \
  --authrpc.vhosts=* \
  --authrpc.port 8551 \
  --nodiscover \
  $HTTP_OPTS \
  $WS_OPTS \
  $MINER_OPTS \
  "$@"

