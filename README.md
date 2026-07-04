# Final Year Project

## Minsinformation Detection Software on X

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
