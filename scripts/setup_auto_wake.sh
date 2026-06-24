#!/bin/bash
# setup_auto_wake.sh — Mac 자동 wake 설정 + 안정성 강화
#
# Usage: sudo bash scripts/setup_auto_wake.sh

set -e

echo "============================================================"
echo "Mac Auto-Wake 설정 — Daily Pipeline 안정성 강화"
echo "============================================================"

if [ "$EUID" -ne 0 ]; then
    echo "✗ sudo 권한 필요. 다시 실행:  sudo bash $0"
    exit 1
fi

echo ""
echo "▸ 1. 매일 06:50 자동 wake 설정"
pmset repeat wakeorpoweron MTWRFSU 6:50:00
echo "  ✓ 완료"

echo ""
echo "▸ 2. AC 전원 연결 시 sleep 안 됨 (전원선 꽂혀있을 때)"
pmset -c sleep 0
pmset -c displaysleep 30
echo "  ✓ AC sleep=0, display sleep=30분"

echo ""
echo "▸ 3. 배터리 모드는 정상 sleep 유지 (배터리 절약)"
pmset -b sleep 15
pmset -b displaysleep 5
echo "  ✓ Battery sleep=15분, display=5분"

echo ""
echo "▸ 4. 자동 로그인 권장 안내"
echo "  System Settings → Users & Groups → Auto Login"
echo "  로그인 비밀번호 필요 시 launchd가 실행 못 함"

echo ""
echo "▸ 5. 현재 설정 확인"
echo "------"
pmset -g sched
echo "------"

echo ""
echo "============================================================"
echo "✓ 설정 완료"
echo "============================================================"
echo ""
echo "주의사항:"
echo "  1) Mac이 SLEEP 상태일 때 wake 가능"
echo "     완전히 SHUTDOWN된 경우 일부 Mac만 자동 부팅 가능"
echo "  2) AC 전원선 연결 권장 (배터리만으로는 불안정)"
echo "  3) 노트북 뚜껑 닫은 상태에서도 wake 가능"
echo "     단, 외부 디스플레이/키보드 미연결 시 일부 Mac은 sleep 유지"
echo ""
echo "제거 명령:  sudo pmset repeat cancel"
echo ""
