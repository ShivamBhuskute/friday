#!/usr/bin/env bash
# Print the PC address to put in the firmware's STREAM_SERVER_IP.
#
# The ESP32 has no discovery: it connects to one hard-coded IP. That value has
# to be this machine's address *on the network the ESP32 joined*, which is not
# necessarily the address on the network you are reading this from. So this
# script also reads back what the firmware currently has, and says plainly
# whether it matches.
set -euo pipefail

# Prefer the wired/wireless address that would actually route to a device on
# the local link, not a docker bridge or a loopback alias.
lan_ip() {
  ip -4 -o addr show scope global 2>/dev/null \
    | awk '$2 !~ /^(docker|br-|veth|virbr)/ { split($4, a, "/"); print a[1]; exit }'
}

IP="$(lan_ip || true)"

if [ -z "$IP" ]; then
  echo "Could not determine a LAN IPv4 address." >&2
  echo "Check the interface is up and connected:  ip -4 addr" >&2
  exit 1
fi

echo "This machine's LAN address:  $IP"
echo
echo "Put this in the firmware:"
echo
echo "    #define STREAM_SERVER_IP  \"$IP\""
echo

# Where the firmware lives, if we can see it.
FIRMWARE="${FRIDAY_FIRMWARE:-}"
if [ -z "$FIRMWARE" ]; then
  for candidate in ../friday_kws/main/friday_kws.cpp \
                   ../friday/main/*.cpp; do
    if [ -f "$candidate" ]; then FIRMWARE="$candidate"; break; fi
  done
fi

if [ -z "$FIRMWARE" ] || [ ! -f "$FIRMWARE" ]; then
  echo "Firmware source not found; skipped the cross-check."
  echo "Point FRIDAY_FIRMWARE at friday_kws.cpp to enable it."
  exit 0
fi

CURRENT="$(sed -n 's/^[[:space:]]*#define[[:space:]]\+STREAM_SERVER_IP[[:space:]]\+"\([^"]*\)".*/\1/p' \
  "$FIRMWARE" | head -1)"

echo "Firmware:  $FIRMWARE"
echo "Currently:  #define STREAM_SERVER_IP \"${CURRENT:-<not found>}\""

if [ "$CURRENT" = "$IP" ]; then
  echo
  echo "Matches. No change needed."
else
  echo
  echo "MISMATCH -- the ESP32 will connect to ${CURRENT:-<nothing>}, not to this machine."
  echo "Update the define, then rebuild and reflash."
  echo
  echo "The ESP32 and this PC must also be on the same subnet, and this port"
  echo "must be reachable:  ingest.port in config.yaml (default 5000), bound to 0.0.0.0."
fi
