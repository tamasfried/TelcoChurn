# Telco Churn MLOps Pipeline

A portfolio project building and deploying a customer churn prediction model end-to-end, in the style of a real MLOps pipeline. The focus is the deployment and monitoring infrastructure around the model, not the model itself, so the model is kept deliberately simple.

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM, via Kaggle) — ~7,000 customers, ~20 mostly categorical features, binary target (`Churn`: Yes/No).

## Stack

- **scikit-learn** — model training (local)
- **FastAPI** — serving the model as an API
- **Docker** — containerising the API
- **Kubernetes** (Minikube/Kind) — local deployment via `kubectl`
- **Cloud Run / Azure Container Apps** — free-tier live demo deployment
- **GitHub Actions** — CI/CD

## Status

Work in progress. Current stage: exploratory data analysis (see `notebooks/`).

- [x] Dataset chosen, project scaffolded
- [x] Initial data exploration (data cleaning, class balance, feature relationships)
- [ ] Baseline model + preprocessing pipeline
- [ ] FastAPI service
- [ ] Docker containerisation
- [ ] Local Kubernetes deployment
- [ ] Logging/monitoring
- [ ] CI/CD via GitHub Actions
- [ ] Live cloud demo


## Local setup

```bash
git clone https://github.com/tamasfried/TelcoChurn.git
cd TelcoChurn
```

**🍎 macOS**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**🪟 Windows (PowerShell)**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
```