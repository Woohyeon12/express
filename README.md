# Tilda Express 도시 배송 최적화

주어진 `orders.csv`, `delivers.csv`를 입력으로 받아 배송 스케줄을 생성하고 `submission.json` 형식으로 저장하는 휴리스틱 솔버입니다.

## 접근 방식

이 문제는 예측보다 `pickup-delivery precedence`, `capacity`, `makespan`을 동시에 만족해야 하는 배차 최적화 문제에 가깝습니다. 그래서 표준 라이브러리만 사용한 휴리스틱 기반 접근을 선택했습니다.

핵심 아이디어는 아래와 같습니다.

1. 누적 배송 시간이 낮은 상위 몇 명 기사를 함께 살펴보고, 이번 배치까지 포함한 `예상 완료 시점`이 가장 좋은 기사를 선택합니다.
2. 선택된 기사 현재 위치 주변에서 처리 가능한 주문을 찾고, 가까운 seed order를 선택합니다. 주변 후보가 모두 소진된 경우에는 전역 fallback 탐색으로 남은 주문 중 가장 적합한 seed를 찾습니다.
3. seed 주변 후보를 모은 뒤, 단순 거리 점수 대신 `정확한 배치 경로 비용 증가량`이 가장 작은 주문을 greedy하게 추가해 같은 배치를 구성합니다.
4. 배치 크기가 작기 때문에, 각 배치 내부에서는 `pickup-delivery precedence`와 `capacity`를 만족하는 최적 action 순서를 exact dynamic programming으로 계산합니다.
5. 초기 해를 만든 뒤, 과부하된 기사 마지막 배치의 일부 주문을 덜 바쁜 기사에게 재배치하는 local rebalancing을 수행합니다.
6. 추가로 두 기사 마지막 배치를 함께 다시 나누는 pairwise repartition을 적용해 makespan을 더 낮춥니다.
7. 모든 주문이 배정되고 rebalancing이 끝날 때까지 반복합니다.

정확해를 찾는 방식은 아니지만, `10,000`건 주문과 `100`명 기사 규모에서도 빠르게 실행되며 항상 제약을 검증하도록 구성했습니다.

## 실행 방법

```bash
python main.py --orders path/to/orders.csv --delivers path/to/delivers.csv --output submission.json --metrics metrics.json
```

예시:

```bash
python main.py --orders "C:\Users\서우현\Desktop\tilda\orders.csv" --delivers "C:\Users\서우현\Desktop\tilda\delivers.csv" --output submission.json --metrics metrics.json
```

## 출력 파일

- `submission.json`: 기사별 action list
- `metrics.json`: 현재 해의 makespan, total distance 등 요약 지표
- `CHANGES.md`: 작업 및 개선 이력

## 제출 형식

`submission.json`은 아래와 같은 구조를 가집니다.

```json
{
  "D001": ["P:O0001", "P:O0042", "D:O0001", "D:O0042"],
  "D002": ["P:O0002", "D:O0002"]
}
```

각 action은 다음 규칙을 따릅니다.

- `P:OrderID`: 주문 픽업
- `D:OrderID`: 주문 배송
