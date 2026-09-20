"""MPS(Metal) 백엔드로 인코딩하는 모델이 여러 개(embeddings.py의 KURE-v1,
reranker.py의 KURE-v2) 있는데, 이들이 프로세스 안에서 서로 다른 스레드로부터
거의 동시에 .encode()를 호출하면 Metal command buffer assertion으로 프로세스
전체가 죽는 문제가 있다 (Abort trap: 6, "A command encoder is already encoding
to this command buffer"). 모델별로 락을 따로 두면 KURE-v1과 KURE-v2가 서로
다른 락을 쓰게 되어 그 둘끼리는 여전히 동시 진입이 가능해지므로 방지가 안 된다.
MPS에 접근하는 모든 인코딩 호출(모델 lazy 생성 포함)이 이 프로세스 전역 락
하나를 공유해서 항상 한 번에 하나의 호출만 MPS에 들어가도록 강제한다."""

import threading

MPS_INFERENCE_LOCK = threading.Lock()