# Final Year Project

## Minsinformation Detection Software on X

This is a Final Year Project Designed and coded by Victor Okafor (w1832950).
---
### How to run project

- To clone this repository run
``` git clone https://github.com/vickleso/W1832950-Prototype.git ```

- When inside the repository run, this will add the secondary petrained model to the project
```git clone https://huggingface.co/roupenminassian/TwHIN-BERT-Misinformation-Classifier ```

- Then in terminal cd into the project directory and activate
```

python3 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

- Once done open another terminal and run:
```
cd frontend
npm init
npm install
npm run dev
```

```
cd backend/app
uvicorn main:app --reload
```

- Open your browser and go to http://localhost:5173/ and have fun :smile:

#### References
- FastAPI - FastAPI. (no date). Available from https://fastapi.tiangolo.com/ [Accessed 4 July 2026].
- First steps | axios | Promise based HTTP client. (no date). Available from https://axios.rest/pages/getting-started/first-steps [Accessed 4 July 2026].
- Hugging Face - Documentation. (no date). Available from https://huggingface.co/docs [Accessed 4 July 2026].
- Paper page - Qwen-VL: A Frontier Large Vision-Language Model with Versatile Abilities. (2023). Available from https://huggingface.co/papers/2308.12966 [Accessed 20 June 2026].
- Popovic, N. (2024). React and Vite. Medium. Available from https://medium.com/@npguapo/react-and-vite-a41b771319f0 [Accessed 21 June 2026].
- Quick Start – React. (no date). Available from https://react.dev/learn [Accessed 21 June 2026].
- roupenminassian/TwHIN-BERT-Misinformation-Classifier · Hugging Face. (2024). Available from https://huggingface.co/roupenminassian/TwHIN-BERT-Misinformation-Classifier [Accessed 20 June 2026].
- Unsloth Docs | Unsloth Documentation. (2026). Available from https://unsloth.ai/docs [Accessed 4 July 2026].
- Xiao, Y. et al. (2025). XFacta: Contemporary, Real-World Dataset and Evaluation for Multimodal Misinformation Detection with Multimodal LLMs. Available from https://doi.org/10.48550/ARXIV.2508.09999 [Accessed 3 November 2025].


