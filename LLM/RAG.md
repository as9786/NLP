# RAG(Retrieval-Augmented Generation, 검색-증강 생성)
- R : 요청된 무엇인가를 집어오는 것
- A : 원래 것에 덧붙이는 것
- LLM을 보다 정확하고 최신 상태로 유지하는데 도움이 되는 framework
- LLM이 자신이 알고 있는 것에만 의존하는 것이 아니라 지식 저장소를 추가
- 저장소의 형태는 internet, 기업의 문서 등 개방적이거나 폐쇄적일 수 있음
- 흐름
1. 사용자의 질의
2. 지식 저장소에 query를 날림
3. Augmented context를 받아옴
4. Prompt + Query + Augmented context를 LLM에 입력
5. 생성된 결과 반환

![image](https://github.com/user-attachments/assets/35b27780-905f-4e94-a8c5-3540b3434357)
