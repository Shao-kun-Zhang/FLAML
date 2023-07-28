"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[7082],{1729:e=>{e.exports=JSON.parse('{"blogPosts":[{"id":"Use flaml.autogen for Local LLMs","metadata":{"permalink":"/FLAML/blog/2023/07/14/Local-LLMs","source":"@site/blog/2023-07-14-Local-LLMs/index.mdx","title":"Use flaml.autogen for Local LLMs","description":"TL;DR:","date":"2023-07-14T00:00:00.000Z","formattedDate":"July 14, 2023","tags":[{"label":"LLM","permalink":"/FLAML/blog/tags/llm"},{"label":"FLAMLv2","permalink":"/FLAML/blog/tags/flam-lv-2"}],"readingTime":2.12,"truncated":false,"authors":[{"name":"Jiale Liu","title":"Undergraduate student at Xidian University","url":"https://leoljl.github.io","imageURL":"https://github.com/LeoLjl/leoljl.github.io/blob/main/profile.jpg?raw=true","key":"jialeliu"}],"nextItem":{"title":"Achieve More, Pay Less - Use GPT-4 Smartly","permalink":"/FLAML/blog/2023/05/18/GPT-adaptive-humaneval"}},"content":"**TL;DR:**\\nWe demonstrate how to use flaml.autogen for local LLM application. As an example, we will initiate an endpoint using [FastChat](https://github.com/lm-sys/FastChat) and perform inference on [ChatGLMv2-6b](https://github.com/THUDM/ChatGLM2-6B).\\n\\n## Preparations\\n\\n### Clone FastChat\\n\\nFastChat provides OpenAI-compatible APIs for its supported models, so you can use FastChat as a local drop-in replacement for OpenAI APIs. However, its code needs minor modification in order to function properly.\\n\\n```bash\\ngit clone https://github.com/lm-sys/FastChat.git\\ncd FastChat\\n```\\n\\n### Download checkpoint\\n\\nChatGLM-6B is an open bilingual language model based on General Language Model (GLM) framework, with 6.2 billion parameters. ChatGLM2-6B is its second-generation version.\\n\\nBefore downloading from HuggingFace Hub, you need to have Git LFS [installed](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage).\\n\\n```bash\\ngit clone https://huggingface.co/THUDM/chatglm2-6b\\n```\\n\\n## Initiate server\\n\\nFirst, launch the controller\\n\\n```bash\\npython -m fastchat.serve.controller\\n```\\n\\nThen, launch the model worker(s)\\n\\n```bash\\npython -m fastchat.serve.model_worker --model-path chatglm2-6b\\n```\\n\\nFinally, launch the RESTful API server\\n\\n```bash\\npython -m fastchat.serve.openai_api_server --host localhost --port 8000\\n```\\n\\nNormally this will work. However, if you encounter error like [this](https://github.com/lm-sys/FastChat/issues/1641), commenting out all the lines containing `finish_reason` in `fastchat/protocol/api_protocal.py` and `fastchat/protocol/openai_api_protocol.py` will fix the problem. The modified code looks like:\\n\\n```python\\nclass CompletionResponseChoice(BaseModel):\\n    index: int\\n    text: str\\n    logprobs: Optional[int] = None\\n    # finish_reason: Optional[Literal[\\"stop\\", \\"length\\"]]\\n\\nclass CompletionResponseStreamChoice(BaseModel):\\n    index: int\\n    text: str\\n    logprobs: Optional[float] = None\\n    # finish_reason: Optional[Literal[\\"stop\\", \\"length\\"]] = None\\n```\\n\\n\\n## Interact with model using `oai.Completion`\\n\\nNow the models can be directly accessed through openai-python library as well as `flaml.oai.Completion` and `flaml.oai.ChatCompletion`.\\n\\n\\n```python\\nfrom flaml import oai\\n\\n# create a text completion request\\nresponse = oai.Completion.create(\\n    config_list=[\\n        {\\n            \\"model\\": \\"chatglm2-6b\\",\\n            \\"api_base\\": \\"http://localhost:8000/v1\\",\\n            \\"api_type\\": \\"open_ai\\",\\n            \\"api_key\\": \\"NULL\\", # just a placeholder\\n        }\\n    ],\\n    prompt=\\"Hi\\",\\n)\\nprint(response)\\n\\n# create a chat completion request\\nresponse = oai.ChatCompletion.create(\\n    config_list=[\\n        {\\n            \\"model\\": \\"chatglm2-6b\\",\\n            \\"api_base\\": \\"http://localhost:8000/v1\\",\\n            \\"api_type\\": \\"open_ai\\",\\n            \\"api_key\\": \\"NULL\\",\\n        }\\n    ],\\n    messages=[{\\"role\\": \\"user\\", \\"content\\": \\"Hi\\"}]\\n)\\nprint(response)\\n```\\n\\nIf you would like to switch to different models, download their checkpoints and specify model path when launching model worker(s).\\n\\n## interacting with multiple local LLMs\\n\\nIf you would like to interact with multiple LLMs on your local machine, replace the `model_worker` step above with a multi model variant:\\n\\n```bash\\npython -m fastchat.serve.multi_model_worker \\\\\\n    --model-path lmsys/vicuna-7b-v1.3 \\\\\\n    --model-names vicuna-7b-v1.3 \\\\\\n    --model-path chatglm2-6b \\\\\\n    --model-names chatglm2-6b\\n```\\n\\nThe inference code would be:\\n\\n```python\\nfrom flaml import oai\\n\\n# create a chat completion request\\nresponse = oai.ChatCompletion.create(\\n    config_list=[\\n        {\\n            \\"model\\": \\"chatglm2-6b\\",\\n            \\"api_base\\": \\"http://localhost:8000/v1\\",\\n            \\"api_type\\": \\"open_ai\\",\\n            \\"api_key\\": \\"NULL\\",\\n        },\\n        {\\n            \\"model\\": \\"vicuna-7b-v1.3\\",\\n            \\"api_base\\": \\"http://localhost:8000/v1\\",\\n            \\"api_type\\": \\"open_ai\\",\\n            \\"api_key\\": \\"NULL\\",\\n        }\\n    ],\\n    messages=[{\\"role\\": \\"user\\", \\"content\\": \\"Hi\\"}]\\n)\\nprint(response)\\n```\\n\\n## For Further Reading\\n\\n* [Documentation](/docs/Use-Cases/Auto-Generation) about `flaml.autogen`\\n* [Documentation](https://github.com/lm-sys/FastChat) about FastChat."},{"id":"Achieve More, Pay Less - Use GPT-4 Smartly","metadata":{"permalink":"/FLAML/blog/2023/05/18/GPT-adaptive-humaneval","source":"@site/blog/2023-05-18-GPT-adaptive-humaneval/index.mdx","title":"Achieve More, Pay Less - Use GPT-4 Smartly","description":"An adaptive way of using GPT-3.5 and GPT-4 outperforms GPT-4 in both coding success rate and inference cost","date":"2023-05-18T00:00:00.000Z","formattedDate":"May 18, 2023","tags":[{"label":"LLM","permalink":"/FLAML/blog/tags/llm"},{"label":"GPT","permalink":"/FLAML/blog/tags/gpt"},{"label":"research","permalink":"/FLAML/blog/tags/research"}],"readingTime":7.73,"truncated":false,"authors":[{"name":"Chi Wang","title":"Principal Researcher at Microsoft Research","url":"https://www.linkedin.com/in/chi-wang-49b15b16/","imageURL":"https://github.com/sonichi.png","key":"sonichi"}],"prevItem":{"title":"Use flaml.autogen for Local LLMs","permalink":"/FLAML/blog/2023/07/14/Local-LLMs"},"nextItem":{"title":"Surpassing 1 Million Downloads - A Retrospective and a Look into the Future","permalink":"/FLAML/blog/2023/05/07/1M-milestone"}},"content":"![An adaptive way of using GPT-3.5 and GPT-4 outperforms GPT-4 in both coding success rate and inference cost](img/humaneval.png)\\n\\n**TL;DR:**\\n* **A case study using the HumanEval benchmark shows that an adaptive way of using multiple GPT models can achieve both much higher accuracy (from 68% to 90%) and lower inference cost (by 18%) than using GPT-4 for coding.**\\n\\n\\nGPT-4 is a big upgrade of foundation model capability, e.g., in code and math, accompanied by a much higher (more than 10x) price per token to use over GPT-3.5-Turbo. On a code completion benchmark, [HumanEval](https://huggingface.co/datasets/openai_humaneval), developed by OpenAI, GPT-4 can successfully solve 68% tasks while GPT-3.5-Turbo does 46%. It is possible to increase the success rate of GPT-4 further by generating multiple responses or making multiple calls. However, that will further increase the cost, which is already nearly 20 times of using GPT-3.5-Turbo and with more restricted API call rate limit. Can we achieve more with less?\\n\\nIn this blog post, we will explore a creative, adaptive way of using GPT models which leads to a big leap forward.\\n\\n## Observations\\n\\n* GPT-3.5-Turbo can alrady solve 40%-50% tasks. For these tasks if we never use GPT-4, we can save nearly 40-50% cost.\\n* If we use the saved cost to generate more responses with GPT-4 for the remaining unsolved tasks, it is possible to solve some more of them while keeping the amortized cost down.\\n\\nThe obstacle of leveraging these observations is that we do not know *a priori* which tasks can be solved by the cheaper model, which tasks can be solved by the expensive model, and which tasks can be solved by paying even more to the expensive model.\\n\\nTo overcome that obstacle, one may want to predict which task requires what model to solve and how many responses are required for each task. Let\'s look at one example code completion task:\\n\\n```python\\ndef vowels_count(s):\\n    \\"\\"\\"Write a function vowels_count which takes a string representing\\n    a word as input and returns the number of vowels in the string.\\n    Vowels in this case are \'a\', \'e\', \'i\', \'o\', \'u\'. Here, \'y\' is also a\\n    vowel, but only when it is at the end of the given word.\\n\\n    Example:\\n    >>> vowels_count(\\"abcde\\")\\n    2\\n    >>> vowels_count(\\"ACEDY\\")\\n    3\\n    \\"\\"\\"\\n```\\n\\nCan we predict whether GPT-3.5-Turbo can solve this task or do we need to use GPT-4? My first guess is that GPT-3.5-Turbo can get it right because the instruction is fairly straightforward. Yet, it turns out that GPT-3.5-Turbo does not consistently get it right, if we only give it one chance. It\'s not obvious (but an interesting research question!) how to predict the performance without actually trying.\\n\\nWhat else can we do? We notice that:\\n**It\'s \\"easier\\" to verify a given solution than finding a correct solution from scratch.**\\n\\nSome simple example test cases are provided in the docstr. If we already have a response generated by a model, we can use those test cases to filter wrong implementations, and either use a more powerful model or generate more responses, until the result passes the example test cases. Moreover, this step can be automated by asking GPT-3.5-Turbo to generate assertion statements from the examples given in the docstr (a simpler task where we can place our bet) and executing the code.\\n\\n## Solution\\n\\nCombining these observations, we can design a solution with two intuitive ideas:\\n\\n* Make use of auto-generated feedback, i.e., code execution results, to filter responses.\\n* Try inference configurations one by one, until one response can pass the filter.\\n\\n![Design](img/design.png)\\n\\nThis solution works adaptively without knowing or predicting which task fits which configuration. It simply tries multiple configurations one by one, starting from the cheapest configuration. Note that one configuration can generate multiple responses (by setting the inference parameter n larger than 1). And different configurations can use the same model and different inference parameters such as n and temperature. Only one response is returned and evaluated per task.\\n\\nAn implementation of this solution is provided in [flaml.autogen](/docs/reference/autogen/code_utils#implement). It uses the following sequence of configurations:\\n\\n1. GPT-3.5-Turbo, n=1, temperature=0\\n1. GPT-3.5-Turbo, n=7, temperature=1, stop=[\\"\\\\nclass\\", \\"\\\\ndef\\", \\"\\\\nif\\", \\"\\\\nprint\\"]\\n1. GPT-4, n=1, temperature=0\\n1. GPT-4, n=2, temperature=1, stop=[\\"\\\\nclass\\", \\"\\\\ndef\\", \\"\\\\nif\\", \\"\\\\nprint\\"]\\n1. GPT-4, n=1, temperature=1, stop=[\\"\\\\nclass\\", \\"\\\\ndef\\", \\"\\\\nif\\", \\"\\\\nprint\\"]\\n\\n## Experiment Results\\n\\nThe first figure in this blog post shows the success rate and average inference cost of the adaptive solution compared with default GPT-4.\\nThe inference cost includes the cost for generating the assertions in our solution. The generated assertions are not always correct, and programs that pass/fail the generated assertions are not always right/wrong. Despite of that, the adaptive solution can increase the success rate (referred to as pass@1 in the literature) from 68% to 90%, while reducing the cost by 18%.\\n\\nHere are a few examples of function definitions which are solved by different configurations in the portfolio.\\n\\n1. Solved by GPT-3.5-Turbo, n=1, temperature=0\\n```python\\ndef compare(game,guess):\\n    \\"\\"\\"I think we all remember that feeling when the result of some long-awaited\\n    event is finally known. The feelings and thoughts you have at that moment are\\n    definitely worth noting down and comparing.\\n    Your task is to determine if a person correctly guessed the results of a number of matches.\\n    You are given two arrays of scores and guesses of equal length, where each index shows a match.\\n    Return an array of the same length denoting how far off each guess was. If they have guessed correctly,\\n    the value is 0, and if not, the value is the absolute difference between the guess and the score.\\n\\n\\n    example:\\n\\n    compare([1,2,3,4,5,1],[1,2,3,4,2,-2]) -> [0,0,0,0,3,3]\\n    compare([0,5,0,0,0,4],[4,1,1,0,0,-2]) -> [4,4,1,0,0,6]\\n    \\"\\"\\"\\n```\\n2. Solved by GPT-3.5-Turbo, n=7, temperature=1, stop=[\\"\\\\nclass\\", \\"\\\\ndef\\", \\"\\\\nif\\", \\"\\\\nprint\\"]: the `vowels_count` function presented earlier.\\n3. Solved by GPT-4, n=1, temperature=0:\\n```python\\ndef string_xor(a: str, b: str) -> str:\\n    \\"\\"\\" Input are two strings a and b consisting only of 1s and 0s.\\n    Perform binary XOR on these inputs and return result also as a string.\\n    >>> string_xor(\'010\', \'110\')\\n    \'100\'\\n    \\"\\"\\"\\n```\\n4. Solved by GPT-4, n=2, temperature=1, stop=[\\"\\\\nclass\\", \\"\\\\ndef\\", \\"\\\\nif\\", \\"\\\\nprint\\"]:\\n```python\\ndef is_palindrome(string: str) -> bool:\\n    \\"\\"\\" Test if given string is a palindrome \\"\\"\\"\\n    return string == string[::-1]\\n\\n\\ndef make_palindrome(string: str) -> str:\\n    \\"\\"\\" Find the shortest palindrome that begins with a supplied string.\\n    Algorithm idea is simple:\\n    - Find the longest postfix of supplied string that is a palindrome.\\n    - Append to the end of the string reverse of a string prefix that comes before the palindromic suffix.\\n    >>> make_palindrome(\'\')\\n    \'\'\\n    >>> make_palindrome(\'cat\')\\n    \'catac\'\\n    >>> make_palindrome(\'cata\')\\n    \'catac\'\\n    \\"\\"\\"\\n```\\n5. Solved by GPT-4, n=1, temperature=1, stop=[\\"\\\\nclass\\", \\"\\\\ndef\\", \\"\\\\nif\\", \\"\\\\nprint\\"]:\\n```python\\ndef sort_array(arr):\\n    \\"\\"\\"\\n    In this Kata, you have to sort an array of non-negative integers according to\\n    number of ones in their binary representation in ascending order.\\n    For similar number of ones, sort based on decimal value.\\n\\n    It must be implemented like this:\\n    >>> sort_array([1, 5, 2, 3, 4]) == [1, 2, 3, 4, 5]\\n    >>> sort_array([-2, -3, -4, -5, -6]) == [-6, -5, -4, -3, -2]\\n    >>> sort_array([1, 0, 2, 3, 4]) [0, 1, 2, 3, 4]\\n    \\"\\"\\"\\n```\\n\\nThe last problem is an example with wrong example test cases in the original definition. It misleads the adaptive solution because a correct implementation is regarded as wrong and more trials are made. The last configuration in the sequence returns the right implementation, even though it does not pass the auto-generated assertions. This example demonstrates that:\\n* Our adaptive solution has a certain degree of fault tolerance.\\n* The success rate and inference cost for the adaptive solution can be further improved if correct example test cases are used.\\n\\nIt is worth noting that the reduced inference cost is the amortized cost over all the tasks. For each individual task, the cost can be either larger or smaller than directly using GPT-4. This is the nature of the adaptive solution: The cost is in general larger for difficult tasks than that for easy tasks.\\n\\nAn example notebook to run this experiment can be found at: https://github.com/microsoft/FLAML/blob/v1.2.1/notebook/research/autogen_code.ipynb\\n\\n## Discussion\\n\\nOur solution is quite simple to [implement](/docs/reference/autogen/code_utils#implement) using a generic interface offered in [`flaml.autogen`](/docs/Use-Cases/Auto-Generation#logic-error), yet the result is quite encouraging.\\n\\nWhile the specific way of generating assertions is application-specific, the main ideas are general in LLM operations:\\n* Generate multiple responses to select - especially useful when selecting a good response is relatively easier than generating a good response at one shot.\\n* Consider multiple configurations to generate responses - especially useful when:\\n  - Model and other inference parameter choice affect the utility-cost tradeoff; or\\n  - Different configurations have complementary effect.\\n\\nA [previous blog post](/blog/2023/04/21/LLM-tuning-math) provides evidence that these ideas are relevant in solving math problems too.\\n`flaml.autogen` uses a technique [EcoOptiGen](https://arxiv.org/abs/2303.04673) to support inference parameter tuning and model selection.\\n\\nThere are many directions of extensions in research and development:\\n* Generalize the way to provide feedback.\\n* Automate the process of optimizing the configurations.\\n* Build adaptive agents for different applications.\\n\\n*Do you find this approach applicable to your use case? Do you have any other challenge to share about LLM applications? Do you like to see more support or research of LLM optimization or automation? Please join our [Discord](https://discord.gg/Cppx2vSPVP) server for discussion.*\\n\\n## For Further Reading\\n\\n* [Documentation](/docs/Use-Cases/Auto-Generation) about `flaml.autogen` and [Research paper](https://arxiv.org/abs/2303.04673).\\n* [Blog post](/blog/2023/04/21/LLM-tuning-math) about a related study for math."},{"id":"Surpassing 1 Million Downloads - A Retrospective and a Look into the Future","metadata":{"permalink":"/FLAML/blog/2023/05/07/1M-milestone","source":"@site/blog/2023-05-07-1M-milestone/index.mdx","title":"Surpassing 1 Million Downloads - A Retrospective and a Look into the Future","description":"TL;DR:","date":"2023-05-07T00:00:00.000Z","formattedDate":"May 7, 2023","tags":[{"label":"LLM","permalink":"/FLAML/blog/tags/llm"},{"label":"LLMOps","permalink":"/FLAML/blog/tags/llm-ops"},{"label":"FLAMLv2","permalink":"/FLAML/blog/tags/flam-lv-2"}],"readingTime":3.66,"truncated":false,"authors":[{"name":"Qingyun Wu","title":"Assistant Professor at the Pennsylvania State University","url":"https://qingyun-wu.github.io/","imageURL":"https://github.com/qingyun-wu.png","key":"qingyunwu"}],"prevItem":{"title":"Achieve More, Pay Less - Use GPT-4 Smartly","permalink":"/FLAML/blog/2023/05/18/GPT-adaptive-humaneval"},"nextItem":{"title":"Does Model and Inference Parameter Matter in LLM Applications? - A Case Study for MATH","permalink":"/FLAML/blog/2023/04/21/LLM-tuning-math"}},"content":"**TL;DR:**\\n* **Celebrating FLAML\'s milestone: 1 million downloads**\\n* **Introducing Large Language Model (LLM) support in the upcoming FLAML v2**\\n\\n\\nThis week, FLAML has reached a significant milestone: 1 million downloads. Originating as an intern research project within Microsoft Research, FLAML has grown into an open-source library used widely across the industry and supported by an active community.\\nAs we celebrate this milestone, we want to recognize the passionate contributors and users who have played an essential role in molding FLAML into the flourishing project it is today. Our heartfelt gratitude goes out to each of you for your unwavering support, constructive feedback, and innovative contributions that have driven FLAML to new heights.\\nA big shoutout to our industrial collaborators from Azure Core, Azure Machine Learning, Azure Synapse Analytics, Microsoft 365, ML.NET, Vowpal Wabbit, Anyscale, Databricks, and Wise; and academic collaborators from MIT, Penn State University, Stevens Institute of Technology, Tel Aviv University, Texas A & M University, University of Manchester, University of Washington, and The Chinese University of Hong Kong etc.\\n\\nWe\'d also like to take the opportunity to reflect on FLAML\'s past achievements and its future roadmap, with a particular focus on large language models (LLM) and LLMOps.\\n\\n## FLAML\'s Journey: Past Achievements and Milestones\\n\\n### Bring AutoML to One\'s Fingertips\\nFLAML offers an off-the-shelf AutoML solution that enables users to quickly discover high-quality models or configurations for common ML/AI tasks. By automatically selecting models and hyperparameters for training or inference, FLAML saves users time and effort. FLAML has significantly reduced development time for developers and data scientists alike, while also providing a convenient way to integrate new algorithms into the pipeline, enabling easy extensions and large-scale parallel tuning. These features make FLAML a valuable tool in R&D efforts for many enterprise users.\\nFLAML is capable of handling a variety of common ML tasks, such as [classification](https://microsoft.github.io/FLAML/docs/Examples/AutoML-Classification), [regression](https://microsoft.github.io/FLAML/docs/Examples/AutoML-Regression), [time series forecasting](https://microsoft.github.io/FLAML/docs/Examples/AutoML-Time%20series%20forecast), [NLP tasks](https://microsoft.github.io/FLAML/docs/Examples/AutoML-Rank), and [generative tasks](https://microsoft.github.io/FLAML/docs/Use-Cases/Auto-Generation), providing a comprehensive solution for various applications.\\n\\n### Speed and Efficiency: The FLAML Advantage\\nWhat sets FLAML apart from other AutoML libraries is its exceptional efficiency, thanks to the economical and efficient hyperparameter optimization and model selection methods developed in our [research](https://microsoft.github.io/FLAML/docs/Research). FLAML is also capable of handling large search spaces with heterogeneous evaluation costs, complex constraints, guidance, and early stopping. The [zero-shot AutoML](https://microsoft.github.io/FLAML/docs/Use-Cases/Zero-Shot-AutoML) option further reduces the cost of AutoML, making FLAML an even more attractive solution for a wide range of applications with low resources.\\n\\n### Easy Customization and Extensibility\\nFLAML is designed for easy extensibility and customization, allowing users to add custom learners, metrics, search space, etc. For example, the support of hierarchical search spaces allows one to first choose an ML learner and then sampling from the hyperparameter space specific to that learner. The level of customization ranges from minimal (providing only training data and task type as input) to full (tuning a user-defined function). This flexibility and support for easy customization have led to FLAML\'s adoption in various domains, including security, finance, marketing, engineering, supply chain, insurance, and healthcare, delivering highly accurate results.\\n\\n## Embracing Large Language Models in FLAML v2\\nAs large language models continue to reshape the AI ecosystem, FLAML is poised to adapt and grow alongside these advancements. Recognizing the importance of large language models, we have recently incorporated an autogen package into FLAML, and are committed to focusing our collective efforts on addressing the unique challenges that arise in LLMOps (Large Language Model Operations).\\n\\nIn its current iteration, FLAML offers support for model selection and inference parameter tuning for large language models. We are actively working on the development of new features, such as low-level inference API with caching, templating, filtering, and higher-level components like LLM-based coding and interactive agents, to enable more effective and economical usage of LLM.\\n\\nWe are eagerly preparing for the launch of FLAML v2, where we will place special emphasis on incorporating and enhancing features specifically tailored for large language models (LLMs), further expanding FLAML\'s capabilities.\\nWe invite contributions from anyone interested in this topic and look forward to collaborating with the community as we shape the future of FLAML and LLMOps together.\\n\\n## For Further Reading\\n\\n* [Documentation about `flaml.autogen`](/docs/Use-Cases/Auto-Generation)\\n* [Code Example: Tune chatGPT for Math Problem Solving with FLAML](https://github.com/microsoft/FLAML/blob/main/notebook/autogen_chatgpt_gpt4.ipynb)\\n\\n*Do you have any experience to share about LLM applications? Do you like to see more support or research of LLMOps? Please join our [Discord](https://discord.gg/Cppx2vSPVP) server for discussion.*"},{"id":"Does Model and Inference Parameter Matter in LLM Applications? - A Case Study for MATH","metadata":{"permalink":"/FLAML/blog/2023/04/21/LLM-tuning-math","source":"@site/blog/2023-04-21-LLM-tuning-math/index.mdx","title":"Does Model and Inference Parameter Matter in LLM Applications? - A Case Study for MATH","description":"level 2 algebra","date":"2023-04-21T00:00:00.000Z","formattedDate":"April 21, 2023","tags":[{"label":"LLM","permalink":"/FLAML/blog/tags/llm"},{"label":"GPT","permalink":"/FLAML/blog/tags/gpt"},{"label":"research","permalink":"/FLAML/blog/tags/research"}],"readingTime":4.97,"truncated":false,"authors":[{"name":"Chi Wang","title":"Principal Researcher at Microsoft Research","url":"https://www.linkedin.com/in/chi-wang-49b15b16/","imageURL":"https://github.com/sonichi.png","key":"sonichi"}],"prevItem":{"title":"Surpassing 1 Million Downloads - A Retrospective and a Look into the Future","permalink":"/FLAML/blog/2023/05/07/1M-milestone"}},"content":"![level 2 algebra](img/level2algebra.png)\\n\\n**TL;DR:**\\n* **Just by tuning the inference parameters like model, number of responses, temperature etc. without changing any model weights or prompt, the baseline accuracy of untuned gpt-4 can be improved by 20% in high school math competition problems.**\\n* **For easy problems, the tuned gpt-3.5-turbo model vastly outperformed untuned gpt-4 in accuracy (e.g., 90% vs. 70%) and cost efficiency. For hard problems, the tuned gpt-4 is much more accurate (e.g., 35% vs. 20%) and less expensive than untuned gpt-4.**\\n* **FLAML can help with model selection, parameter tuning, and cost-saving in LLM applications.**\\n\\n\\nLarge language models (LLMs) are powerful tools that can generate natural language texts for various applications, such as chatbots, summarization, translation, and more. GPT-4 is currently the state of the art LLM in the world. Is model selection irrelevant? What about inference parameters?\\n\\nIn this blog post, we will explore how model and inference parameter matter in LLM applications, using a case study for [MATH](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/be83ab3ecd0db773eb2dc1b0a17836a1-Abstract-round2.html), a benchmark for evaluating LLMs on advanced mathematical problem solving. MATH consists of 12K math competition problems from AMC-10, AMC-12 and AIME. Each problem is accompanied by a step-by-step solution.\\n\\nWe will use the new subpackage [`flaml.autogen`](docs/Use-Cases/Auto-Generation) to automatically find the best model and inference parameter for LLMs on a given task and dataset given an inference budget, using a novel low-cost search & pruning strategy. FLAML currently supports all the LLMs from OpenAI, such as GPT-3.5 and GPT-4.\\n\\nWe will use FLAML to perform model selection and inference parameter tuning. Then we compare the performance and inference cost on solving algebra problems with the untuned gpt-4. We will also analyze how different difficulty levels affect the results.\\n\\n## Experiment Setup\\n\\nWe use FLAML to select between the following models with a target inference budget $0.02 per instance:\\n- gpt-3.5-turbo, a relatively cheap model that powers the popular ChatGPT app\\n- gpt-4, the state of the art LLM that costs more than 10 times of gpt-3.5-turbo\\n\\nWe adapt the models using 20 examples in the train set, using the problem statement as the input and generating the solution as the output. We use the following inference parameters:\\n\\n- temperature: The parameter that controls the randomness of the output text. A higher temperature means more diversity but less coherence. We search for the optimal temperature in the range of [0, 1].\\n- top_p: The parameter that controls the probability mass of the output tokens. Only tokens with a cumulative probability less than or equal to top-p are considered. A lower top-p means more diversity but less coherence. We search for the optimal top-p in the range of [0, 1].\\n- max_tokens: The maximum number of tokens that can be generated for each output. We search for the optimal max length in the range of [50, 1000].\\n- n: The number of responses to generate. We search for the optimal n in the range of [1, 100].\\n- prompt: We use the template: \\"{problem} Solve the problem carefully. Simplify your answer as much as possible. Put the final answer in \\\\\\\\boxed{{}}.\\" where {problem} will be replaced by the math problem instance.\\n\\nIn this experiment, when n > 1, we find the answer with highest votes among all the responses and then select it as the final answer to compare with the ground truth. For example, if n = 5 and 3 of the responses contain a final answer 301 while 2 of the responses contain a final answer 159, we choose 301 as the final answer. This can help with resolving potential errors due to randomness. We use the average accuracy and average inference cost as the metric to evaluate the performance over a dataset. The inference cost of a particular instance is measured by the price per 1K tokens and the number of tokens consumed.\\n\\n## Experiment Results\\n\\nThe first figure in this blog post shows the average accuracy and average inference cost of each configuration on the level 2 Algebra test set.\\n\\nSurprisingly, the tuned gpt-3.5-turbo model is selected as a better model and it vastly outperforms untuned gpt-4 in accuracy (92% vs. 70%) with equal or 2.5 times higher inference budget.\\nThe same observation can be obtained on the level 3 Algebra test set.\\n\\n![level 3 algebra](img/level3algebra.png)\\n\\nHowever, the selected model changes on level 4 Algebra.\\n\\n![level 4 algebra](img/level4algebra.png)\\n\\nThis time gpt-4 is selected as the best model. The tuned gpt-4 achieves much higher accuracy (56% vs. 44%) and lower cost than the untuned gpt-4.\\nOn level 5 the result is similar.\\n\\n![level 5 algebra](img/level5algebra.png)\\n\\nWe can see that FLAML has found different optimal model and inference parameters for each subset of a particular level, which shows that these parameters matter in cost-sensitive LLM applications and need to be carefully tuned or adapted.\\n\\nAn example notebook to run these experiments can be found at: https://github.com/microsoft/FLAML/blob/v1.2.1/notebook/autogen_chatgpt.ipynb\\n\\n## Analysis and Discussion\\n\\nWhile gpt-3.5-turbo demonstrates competitive accuracy with voted answers in relatively easy algebra problems under the same inference budget, gpt-4 is a better choice for the most difficult problems. In general, through parameter tuning and model selection, we can identify the opportunity to save the expensive model for more challenging tasks, and improve the overall effectiveness of a budget-constrained system.\\n\\nThere are many other alternative ways of solving math problems, which we have not covered in this blog post. When there are choices beyond the inference parameters, they can be generally tuned via [`flaml.tune`](docs/Use-Cases/Tune-User-Defined-Function).\\n\\nThe need for model selection, parameter tuning and cost saving is not specific to the math problems. The [Auto-GPT](https://github.com/Significant-Gravitas/Auto-GPT) project is an example where high cost can easily prevent a generic complex task to be accomplished as it needs many LLM inference calls.\\n\\n## For Further Reading\\n\\n* [Research paper about the tuning technique](https://arxiv.org/abs/2303.04673)\\n* [Documentation about `flaml.autogen`](/docs/Use-Cases/Auto-Generation)\\n\\n*Do you have any experience to share about LLM applications? Do you like to see more support or research of LLM optimization or automation? Please join our [Discord](https://discord.gg/Cppx2vSPVP) server for discussion.*"}]}')}}]);