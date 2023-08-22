---
title: LLM Security
date: 2022-09-24
toc: false

image:
  filename: covers/pexels-nuno-fangueiro-12125258.jpg
  caption:  Monstera - Nuno Fangueiro
---

LLM security is the investigation of the failure modes of LLMs in use, the conditions that lead to them, and their mitigations.

Here are links to large language model security content - research, papers, and news - posted by [@llm_sec](https://twitter.com/llm_sec)

Got a tip/link? Open a [pull request](https://github.com/leondz/llmsec-site) or send a [DM](https://twitter.com/llm_sec).

## Attacks


### Adversarial

* [A LLM Assisted Exploitation of AI-Guardian](https://arxiv.org/abs/2307.15008)
* [Adversarial Examples Are Not Bugs, They Are Features](https://arxiv.org/abs/1905.02175) 🌶️
* [Are Aligned Language Models “Adversarially Aligned”?](https://www.youtube.com/watch?v=uqOfC3KSZFc) 🌶️
* [Bad Characters: Imperceptible NLP Attacks](https://arxiv.org/abs/2106.09898)
* [Expanding Scope: Adapting English Adversarial Attacks to Chinese](https://aclanthology.org/2023.trustnlp-1.24/)
* [Gradient-based Adversarial Attacks against Text Transformers](https://arxiv.org/abs/2104.13733)
* [Gradient-Based Word Substitution for Obstinate Adversarial Examples Generation in Language Models](https://arxiv.org/abs/2307.12507)

### Backdoors & data poisoning

* [A backdoor attack against LSTM-based text classification systems](https://arxiv.org/abs/1905.12457) "Submitted on 29 May 2019"!
* [A Gradient Control Method for Backdoor Attacks on Parameter-Efficient Tuning](https://aclanthology.org/2023.acl-long.194/)
* [Backdooring Neural Code Search](https://arxiv.org/abs/2305.17506) 🌶️
* [Be Careful about Poisoned Word Embeddings: Exploring the Vulnerability of the Embedding Layers in NLP Models](https://arxiv.org/abs/2103.15543)
* [BadNL: Backdoor Attacks against NLP Models with Semantic-preserving Improvements](https://arxiv.org/abs/2006.01043)
* [BITE: Textual Backdoor Attacks with Iterative Trigger Injection](https://arxiv.org/abs/2205.12700) 🌶️
* [Hidden Killer: Invisible Textual Backdoor Attacks with Syntactic Trigger](https://arxiv.org/abs/2105.12400) 🌶️
* [Instructions as Backdoors: Backdoor Vulnerabilities of Instruction Tuning for Large Language Models](https://arxiv.org/abs/2305.14710)
* [On the Exploitability of Instruction Tuning](https://arxiv.org/abs/2306.17194)
* [Poisoning Web-Scale Training Datasets is Practical](https://arxiv.org/abs/2302.10149) 🌶️

### Prompt injection

* [Bing Chat: Data Exfiltration Exploit Explained](https://embracethered.com/blog/posts/2023/bing-chat-data-exfiltration-poc-and-fix/) 🌶️
* [Compromising LLMs: The Advent of AI Malware](https://www.blackhat.com/us-23/briefings/schedule/index.html#compromising-llms-the-advent-of-ai-malware-33075)
* [GPT-4 Is Too Smart To Be Safe: Stealthy Chat with LLMs via Cipher](https://arxiv.org/abs/2308.06463)
* [Hackers Compromised ChatGPT Model with Indirect Prompt Injection](https://gbhackers.com/hackers-compromised-chatgpt-model/)
* [Ignore Previous Prompt: Attack Techniques For Language Models](https://arxiv.org/abs/2211.09527) 🌶️
* [Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection](https://arxiv.org/abs/2302.12173) 🌶️
* [Prompt Injection attack against LLM-integrated Applications](https://arxiv.org/abs/2306.05499)
* [Safeguarding Crowdsourcing Surveys from ChatGPT with Prompt Injection](https://arxiv.org/abs/2306.08833)
* [Virtual Prompt Injection for Instruction-Tuned Large Language Models](https://arxiv.org/abs/2307.16888)

### Jailbreaking

* ["Do Anything Now": Characterizing and Evaluating In-The-Wild Jailbreak Prompts on Large Language Models](https://arxiv.org/abs/2308.03825)
* [JAILBREAKER: Automated Jailbreak Across Multiple Large Language Model Chatbots](https://arxiv.org/pdf/2307.08715.pdf)
* [Jailbroken: How Does LLM Safety Training Fail?](https://arxiv.org/abs/2307.02483)
* [LLM Censorship: A Machine Learning Challenge Or A Computer Security Problem?](https://www.cl.cam.ac.uk/~is410/Papers/llm_censorship.pdf) (mosaic prompts)

### Data extraction

* [Extracting Training Data from Large Language Models](https://arxiv.org/abs/2012.07805)
* [Prompts Should not be Seen as Secrets: Systematically Measuring Prompt Extraction Attack Success](https://arxiv.org/abs/2307.06865)
* [ProPILE: Probing Privacy Leakage in Large Language Models](https://arxiv.org/abs/2307.01881) 🌶️
* [Training Data Extraction From Pre-trained Language Models: A Survey](https://aclanthology.org/2023.trustnlp-1.23/)

### Denial of service

* [Sponge Examples: Energy-Latency Attacks on Neural Networks](https://arxiv.org/abs/2006.03463) 🌶️

### Escalation

* [Hacking Auto-GPT and escaping its docker container](https://positive.security/blog/auto-gpt-rce)

### XSS/CSRF/CPRF

* [LLM causing self-XSS](https://hackstery.com/2023/07/10/llm-causing-self-xss/)

### Cross-model

* [Exploring the Vulnerability of Natural Language Processing Models via Universal Adversarial Texts](https://aclanthology.org/2021.alta-1.14/)

### Multimodal

* [(Ab)using Images and Sounds for Indirect Instruction Injection in Multi-Modal LLMs](https://arxiv.org/abs/2307.10490)
* [Image to Prompt Injection with Google Bard](https://embracethered.com/blog/posts/2023/google-bard-image-to-prompt-injection/)
* [Plug and Pray: Exploiting off-the-shelf components of Multi-Modal Models](https://arxiv.org/abs/2307.14539)
* [Visual Adversarial Examples Jailbreak Aligned Large Language Models](https://arxiv.org/abs/2306.13213)


### Model theft

* [Stealing Machine Learning Models via Prediction APIs](https://arxiv.org/abs/1609.02943)

### Attack automation

* [FakeToxicityPrompts: Automatic Red Teaming](https://interhumanagreement.substack.com/p/faketoxicityprompts-automatic-red)
* [FLIRT: Feedback Loop In-context Red Teaming](https://huggingface.co/papers/2308.04265)
* [Red Teaming Language Models with Language Models](https://arxiv.org/abs/2202.03286)
* [Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned](https://arxiv.org/abs/2209.07858)

## Defenses & Detections

### against things other than backdoors

* [Defending ChatGPT against Jailbreak Attack via Self-Reminder](https://assets.researchsquare.com/files/rs-2873090/v1_covered_3dc9af48-92ba-491e-924d-b13ba9b7216f.pdf?c=1686882819)
* [Diffusion Theory as a Scalpel: Detecting and Purifying Poisonous Dimensions in Pre-trained Language Models Caused by Backdoor or Bias](https://arxiv.org/abs/2305.04547)
* [FedMLSecurity: A Benchmark for Attacks and Defenses in Federated Learning and LLMs](https://arxiv.org/abs/2306.04959)
* [Interpretability and Transparency-Driven Detection and Transformation of Textual Adversarial Examples (IT-DT)](https://arxiv.org/abs/2307.01225)
* [LLM Self Defense: By Self Examination, LLMs Know They Are Being Tricked](https://arxiv.org/abs/2308.07308)
* [Make Text Unlearnable: Exploiting Effective Patterns to Protect Personal Data](https://aclanthology.org/2023.trustnlp-1.22/)
* [Mitigating Stored Prompt Injection Attacks Against LLM Applications](https://developer.nvidia.com/blog/mitigating-stored-prompt-injection-attacks-against-llm-applications/?utm_source=tldrsec.com&utm_medium=referral&utm_campaign=tl-dr-sec-194-cnappgoat-kubefuzz-tl-dr-sec-swag)
* [Secure your machine learning with Semgrep](https://blog.trailofbits.com/2022/10/03/semgrep-maching-learning-static-analysis/)
* [Sparse Logits Suffice to Fail Knowledge Distillation](https://openreview.net/forum?id=BxZgduuNDl5)
* [Text-CRS: A Generalized Certified Robustness Framework against Textual Adversarial Attacks](https://arxiv.org/abs/2307.16630)
* [Towards building a robust toxicity predictor](https://www.amazon.science/publications/towards-building-a-robust-toxicity-predictor)

### against backdoors / backdoor insertion

* [Defending against Insertion-based Textual Backdoor Attacks via Attribution](https://aclanthology.org/2023.findings-acl.561/)
* [GPTs Don’t Keep Secrets: Searching for Backdoor Watermark Triggers in Autoregressive Language Models](https://aclanthology.org/2023.trustnlp-1.21/) 🌶️
* [IMBERT: Making BERT Immune to Insertion-based Backdoor Attacks](https://aclanthology.org/2023.trustnlp-1.25/) 🌶️
* [Maximum Entropy Loss, the Silver Bullet Targeting Backdoor Attacks in Pre-trained Language Models](https://aclanthology.org/2023.findings-acl.237/)
* [ONION: A Simple and Effective Defense Against Textual Backdoor Attacks](https://arxiv.org/abs/2011.10369)
* [ParaFuzz: An Interpretability-Driven Technique for Detecting Poisoned Samples in NLP](https://arxiv.org/abs/2308.02122) 🌶️


## Evaluation

* [PromptBench: Towards Evaluating the Robustness of Large Language Models on Adversarial Prompts](https://arxiv.org/abs/2306.04528)

## Practices

* [A framework to securely use LLMs in companies - Part 1: Overview of Risks](https://boringappsec.substack.com/p/edition-21-a-framework-to-securely)
* [All the Hard Stuff Nobody Talks About when Building Products with LLMs](https://www.honeycomb.io/blog/hard-stuff-nobody-talks-about-llm)
* [Artificial intelligence and machine learning security](https://learn.microsoft.com/en-us/security/engineering/failure-modes-in-machine-learning) (microsoft) 🌶️
* [Assessing Language Model Deployment with Risk Cards](https://arxiv.org/abs/2303.18190)
* [Protect Your Prompts: Protocols for IP Protection in LLM Applications](https://arxiv.org/abs/2306.06297)
* ["Real Attackers Don't Compute Gradients": Bridging the Gap Between Adversarial ML Research and Practice](https://arxiv.org/abs/2212.14315) 🌶️
* [Securing LLM Systems Against Prompt Injection](https://developer.nvidia.com/blog/securing-llm-systems-against-prompt-injection/)
* [Threat Modeling LLM Applications](https://aivillage.org/large%20language%20models/threat-modeling-llm/)
* [Toward Comprehensive Risk Assessments and Assurance of AI-Based Systems](https://docs.google.com/viewer?url=https://raw.githubusercontent.com/trailofbits/publications/master/papers/toward_comprehensive_risk_assessments.pdf)
* [Understanding the risks of deploying LLMs in your enterprise](https://www.moveworks.com/insights/risks-of-deploying-llms-in-your-enterprise)

## Analyses & surveys

* [Latent Jailbreak: A Test Suite for Evaluating Both Text Safety and Output Robustness of Large Language Models](https://arxiv.org/abs/2307.08487) 🌶️
* [Chatbots to ChatGPT in a Cybersecurity Space: Evolution, Vulnerabilities, Attacks, Challenges, and Future Recommendations](https://arxiv.org/abs/2306.09255)
* [OWASP Top 10 for LLM vulnerabilities](https://llmtop10.com/) 🌶️
* [Towards Security Threats of Deep Learning Systems: A Survey](https://arxiv.org/abs/1911.12562)

## Policy, legal, ethical, and social

* [Cybercrime and Privacy Threats of Large Language Models](https://ieeexplore.ieee.org/abstract/document/10174273)
* [Ethical Considerations and Policy Implications for Large Language Models: Guiding Responsible Development and Deployment](https://arxiv.org/abs/2308.02678)
* [Frontier AI Regulation: Managing Emerging Risks to Public Safety](https://arxiv.org/abs/2307.03718)
* [Loose-lipped large language models spill your secrets: The privacy implications of large language models](https://jolt.law.harvard.edu/assets/articlePDFs/v36/Winograd-Loose-Lipped-LLMs.pdf)
* [On the Trustworthiness Landscape of State-of-the-art Generative Models: A Comprehensive Survey](https://arxiv.org/abs/2307.16680)
* [On the Dangers of Stochastic Parrots: Can Language Models Be Too Big? 🦜](https://dl.acm.org/doi/10.1145/3442188.3445922) 🌶️
* [Product Liability for Defective AI](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4515202)
* [The last attempted AI revolution in security, and the next one](https://drive.google.com/file/d/1BbSIBayQ1RHVSnh-FnaeXr8xjw5SVJV8/view?pli=1)
* [Unveiling Security, Privacy, and Ethical Concerns of ChatGPT](https://arxiv.org/abs/2307.14192)

## Software

* [BITE](https://github.com/INK-USC/BITE) Textual Backdoor Attacks with Iterative Trigger Injection
* [garak](https://github.com/leondz/garak/) LLM vulnerability scanner 🌶️🌶️
* [rebuff](https://github.com/protectai/rebuff) LLM Prompt Injection Detector


### Generic MLsec

* [Adversarial Robustness Toolkit](https://github.com/Trusted-AI/adversarial-robustness-toolbox)
* [nvtrust](https://github.com/NVIDIA/nvtrust) Ancillary open source software to support confidential computing on NVIDIA GPUs

</hr>

🌶️ = extra spicy