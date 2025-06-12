# Lang chain

- LangChain은 LLM을 사용하는 application development을 위한 open source framework로, python과 javascript library 제공
- ODBC(Open Database Connectivity)와 동작하는 개념이 유사

## 1. 구성 요소

### LLM abstraction
- 추상화를 통해 LLM application programming을 간소화
- 사용자에게 불필요한 세부 사항을 숨겨 복잡성을 처리
- 사용자는 모든 것을 이해하지 않아도 자신의 code를 구현 가능
- LangChain abstraction은 언어 모형을 사용하는 데 필요한 일반적인 단계와 개념

### Prompts
- LangChain은 모형에 명령을 잘 전달하기 위한 prompt template class가 존재
- LangChain prompt는 context and query를 수동으로 작성할 필요 X

### Chains
- 연결 고리를 만드는 것
- LLM을 다른 구성 요소와 결합하여 일련의 작업을 실행함으로써 application을 생성

### Indexes
- Index = 외부 data
1. Document loaders
  - Load data source
2. Vector database
  - Vector database는 data point를 vector embedding으로 변환. 고정된 수의 차원을 가진 vector representation으로 유사성을 나타내기에 매우 효율적인 검색 수단
3. Text splitters
4. Memory
  - 사용자와의 대화를 기억하고 향후 상호 작용에 해당 정보를 적용
  - 대화 전체를 기억하거나 요약을 기억
5. Agents
  - Example : Chatbot
