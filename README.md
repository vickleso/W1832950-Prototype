# Final Year Project

## Minsinformation Detection Software on X

---
### How to run project

- To clone this repository run
``` git clone https://github.com/vickleso/Final-Year-Project.git ```

- Then in terminal cd into the project directory and activate
```
cd Final-Year-Project
python3 -m venv .env
.env\Scripts\activate
pip install -r requirements.txt
```

- Once done open another terminal (personal reccommendation) and run:
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
