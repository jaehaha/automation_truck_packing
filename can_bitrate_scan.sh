#!/bin/bash
# 여러 CAN 속도로 2초씩 듣기만 하면서(송신 없음) 로봇팔 프레임이 잡히는 속도를 찾는다.
# 사용: sudo bash can_bitrate_scan.sh [can0]
IFACE="${1:-can0}"
DIR="$(cd "$(dirname "$0")" && pwd)"

for BR in 1000000 500000 250000 125000; do
    ip link set "$IFACE" down
    ip link set "$IFACE" up type can bitrate "$BR" || { echo "$BR: 설정 실패"; continue; }
    "$DIR/.venv/bin/python" - "$IFACE" "$BR" <<'EOF'
import sys, time, collections, can
iface, br = sys.argv[1], sys.argv[2]
bus = can.Bus(channel=iface, interface="socketcan")
ids = collections.Counter()
end = time.time() + 2
while time.time() < end:
    m = bus.recv(0.2)
    if m is not None and not m.is_error_frame:
        ids[hex(m.arbitration_id)] += 1
bus.shutdown()
print(f"bitrate {br:>7}: 프레임 {sum(ids.values())}개  ID {sorted(ids)[:8]}")
EOF
done

# 끝나면 Piper 기본 속도로 되돌려 둔다.
ip link set "$IFACE" down
ip link set "$IFACE" up type can bitrate 1000000
echo "완료: $IFACE 를 1000000 으로 되돌렸습니다."
