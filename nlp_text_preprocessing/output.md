output in **Markdown**:


### Cleaned Text (No Punctuation, No Numbers):
```plaintext
Hey  I just found this cool website  httpswwwexamplecom              that has amazing articles on AI  with  articles  learning resources Also check out this wwwaiworldcom I think johndoe clarasmith should check it out Its a great read especially for tech enthusiasts like him Have you seen the latest article by Dr. Alice Smith on the future of robotics  I cant wait to learn more  Also dont forget to follow their social media for updates artificialintelligence nextgenai hopeai genai aiworld 
```
---

### Cleaned Text (No Extra Spaces):
```plaintext
Hey I just found this cool website httpswwwexamplecom that has amazing articles on AI with articles learning resources Also check out this wwwaiworldcom I think johndoe clarasmith should check it out Its a great read especially for tech enthusiasts like him Have you seen the latest article by Dr. Alice Smith on the future of robotics I cant wait to learn more Also dont forget to follow their social media for updates artificialintelligence nextgenai hopeai genai aiworld
```
---

### Cleaned Text (After Removing Emojis, URLs, Mentions):
```plaintext
Hey I just found this cool website  that has amazing articles on AI with articles learning resources Also check out this  I think johndoe clarasmith should check it out Its a great read especially for tech enthusiasts like him Have you seen the latest article by Dr. Alice Smith on the future of robotics I cant wait to learn more Also dont forget to follow their social media for updates artificialintelligence nextgenai hopeai genai aiworld
```
---

### Lowercase Sentences:
```plaintext
hey i just found this cool website  that has amazing articles on ai with articles learning resources also check out this  i think johndoe clarasmith should check it out its a great read especially for tech enthusiasts like him have you seen the latest article by dr alice smith on the future of robotics i cant wait to learn more also dont forget to follow their social media for updates artificialintelligence nextgenai hopeai genai aiworld
```
---

### Sentence Tokens:
`['"Hey!', '👋 I just found this cool website 📱: https://www.example.com                                                                                                that has amazing articles on AI 🤖 with 100 articles 30 learning resources.', 'Also check out this www.aiworld.com.', 'I think @john_doe @clara_smith002 should check it out!', 'It’s a great read, especially for tech enthusiasts like him.', 'Have you seen the latest article by Dr. Alice Smith on the future of robotics?', "🚀 I can't wait to learn more!", '🔍 Also, don’t forget to follow their social media for updates #artificialintelligence #nextgenai #hopeai #genai #aiworld 🌐.']`

---

### Word Tokens:
`['hey', 'i', 'just', 'found', 'this', 'cool', 'website', 'that', 'has', 'amazing', 'articles', 'on', 'ai', 'with', 'articles', 'learning', 'resources', 'also', 'check', 'out', 'this', 'i', 'think', 'johndoe', 'clarasmith', 'should', 'check', 'it', 'out', 'its', 'a', 'great', 'read', 'especially', 'for', 'tech', 'enthusiasts', 'like', 'him', 'have', 'you', 'seen', 'the', 'latest', 'article', 'by', 'dr', 'alice', 'smith', 'on', 'the', 'future', 'of', 'robotics', 'i', 'cant', 'wait', 'to', 'learn', 'more', 'also', 'dont', 'forget', 'to', 'follow', 'their', 'social', 'media', 'for', 'updates', 'artificialintelligence', 'nextgenai', 'hopeai', 'genai', 'aiworld']`

---

### Filtered Words (After Stopword Removal):
`['hey', 'found', 'cool', 'website', 'amazing', 'articles', 'ai', 'articles', 'learning', 'resources', 'also', 'check', 'think', 'johndoe', 'clarasmith', 'check', 'great', 'read', 'especially', 'tech', 'enthusiasts', 'like', 'seen', 'latest', 'article', 'dr', 'alice', 'smith', 'future', 'robotics', 'cant', 'wait', 'learn', 'also', 'dont', 'forget', 'follow', 'social', 'media', 'updates', 'artificialintelligence', 'nextgenai', 'hopeai', 'genai', 'aiworld']`

---

### Stemming Process:
- hey  -->  hey
- found  -->  found
- cool  -->  cool
- website  -->  websit
- amazing  -->  amaz
- articles  -->  articl
- ai  -->  ai
- articles  -->  articl
- learning  -->  learn
- resources  -->  resourc
- also  -->  also
- check  -->  check
- think  -->  think
- johndoe  -->  johndo
- clarasmith  -->  clarasmith
- check  -->  check
- great  -->  great
- read  -->  read
- especially  -->  especi
- tech  -->  tech
- enthusiasts  -->  enthusiast
- like  -->  like
- seen  -->  seen
- latest  -->  latest
- article  -->  articl
- dr  -->  dr
- alice  -->  alic
- smith  -->  smith
- future  -->  futur
- robotics  -->  robot
- cant  -->  cant
- wait  -->  wait
- learn  -->  learn
- also  -->  also
- dont  -->  dont
- forget  -->  forget
- follow  -->  follow
- social  -->  social
- media  -->  media
- updates  -->  updat
- artificialintelligence  -->  artificialintellig
- nextgenai  -->  nextgenai
- hopeai  -->  hopeai
- genai  -->  genai
- aiworld  -->  aiworld

**Stemming words**:
`['hey', 'found', 'cool', 'websit', 'amaz', 'articl', 'ai', 'articl', 'learn', 'resourc', 'also', 'check', 'think', 'johndo', 'clarasmith', 'check', 'great', 'read', 'especi', 'tech', 'enthusiast', 'like', 'seen', 'latest', 'articl', 'dr', 'alic', 'smith', 'futur', 'robot', 'cant', 'wait', 'learn', 'also', 'dont', 'forget', 'follow', 'social', 'media', 'updat', 'artificialintellig', 'nextgenai', 'hopeai', 'genai', 'aiworld']`

---

### Lemmatization Process:
- hey  -->  hey
- found  -->  found
- cool  -->  cool
- website  -->  website
- amazing  -->  amazing
- articles  -->  article
- ai  -->  ai
- articles  -->  article
- learning  -->  learning
- resources  -->  resource
- also  -->  also
- check  -->  check
- think  -->  think
- johndoe  -->  johndoe
- clarasmith  -->  clarasmith
- check  -->  check
- great  -->  great
- read  -->  read
- especially  -->  especially
- tech  -->  tech
- enthusiasts  -->  enthusiast
- like  -->  like
- seen  -->  seen
- latest  -->  latest
- article  -->  article
- dr  -->  dr
- alice  -->  alice
- smith  -->  smith
- future  -->  future
- robotics  -->  robotics
- cant  -->  cant
- wait  -->  wait
- learn  -->  learn
- also  -->  also
- dont  -->  dont
- forget  -->  forget
- follow  -->  follow
- social  -->  social
- media  -->  medium
- updates  -->  update
- artificialintelligence  -->  artificialintelligence
- nextgenai  -->  nextgenai
- hopeai  -->  hopeai
- genai  -->  genai
- aiworld  -->  aiworld

**Lemmatized words**:
`['hey', 'found', 'cool', 'website', 'amazing', 'article', 'ai', 'article', 'learning', 'resource', 'also', 'check', 'think', 'johndoe', 'clarasmith', 'check', 'great', 'read', 'especially', 'tech', 'enthusiast', 'like', 'seen', 'latest', 'article', 'dr', 'alice', 'smith', 'future', 'robotics', 'cant', 'wait', 'learn', 'also', 'dont', 'forget', 'follow', 'social', 'medium', 'update', 'artificialintelligence', 'nextgenai', 'hopeai', 'genai', 'aiworld']`

---

### POS Tagging Results:
`[('hey', 'NN'), ('found', 'VBD'), ('cool', 'JJ'), ('website', 'RB'), ('amazing', 'JJ'), ('article', 'NN'), ('ai', 'NN'), ('article', 'NN'), ('learning', 'VBG'), ('resource', 'NN'), ('also', 'RB'), ('check', 'VB'), ('think', 'NN'), ('johndoe', 'NN'), ('clarasmith', 'NN'), ('check', 'NN'), ('great', 'JJ'), ('read', 'JJ'), ('especially', 'RB'), ('tech', 'JJ'), ('enthusiast', 'NN'), ('like', 'IN'), ('seen', 'VBN'), ('latest', 'JJS'), ('article', 'NN'), ('dr', 'NN'), ('alice', 'NN'), ('smith', 'JJ'), ('future', 'NN'), ('robotics', 'NNS'), ('cant', 'JJ'), ('wait', 'NN'), ('learn', 'NN'), ('also', 'RB'), ('dont', 'VBZ'), ('forget', 'VB'), ('follow', 'JJ'), ('social', 'JJ'), ('medium', 'NN'), ('update', 'JJ'), ('artificialintelligence', 'NN'), ('nextgenai', 'JJ'), ('hopeai', 'NN'), ('genai', 'NN'), ('aiworld', 'NN')]`

---

### NER Results (Using POS Tagged Words):

```plaintext
(S
  hey
  found
  cool
  website
  amazing
  article
  ai
  article
  learning
  resource
  also
  check
  think
  johndoe
  clarasmith
  check
  great
  read
  especially
  tech
  enthusiast
  like
  seen
  latest
  article
  dr
  alice
  smith
  future
  robotics
  cant
  wait
  learn
  also
  dont
  forget
  follow
  social
  medium
  update
  artificialintelligence
  nextgenai
  hopeai
  genai
  aiworld)
```


---
