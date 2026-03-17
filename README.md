# EasyMoney_Capstone

## Business Problem
EasyMoney es una plataforma digital de comercialización de productos financieros fundada por Carol Denver, cuyo objetivo es ofrecer a los clientes soluciones financieras sencillas como ahorro, inversión y tarjetas mediante una interfaz digital fácil de usar. Su primer producto, una cuenta de ahorro basada en el redondeo automático de compras, tuvo gran éxito y permitió a la empresa ampliar su oferta de servicios financieros en alianza con easyBanking S.A., entidad supervisada por el Banco de España. Gracias a varias rondas de financiación lideradas por inversores como Lion Global Management, la empresa logró crecer rápidamente y captar una amplia base de clientes; sin embargo, actualmente enfrenta una situación crítica debido a que los fondos obtenidos se están agotando y aún no alcanza rentabilidad, por lo que necesita incrementar las ventas y mejorar la rentabilidad de su cartera actual de clientes, siendo una posible solución el desarrollo de un modelo de predicción que permita identificar qué clientes tienen mayor probabilidad de comprar un producto financiero, optimizando así las campañas comerciales y aumentando la conversión.

---

## Objective
- Analizar y unificar el dataset para conocer los productos y elegir un target.
- Comparar diferentes modelos de prospeccion de compra de un producto.
- Evaluar la prediccion de mi modelo.

---

## Dataset
- Source: interna
- Granularity: Diaria
- Period: 2018–2019
- Dataset: 
  - `customer_commercial_activity` → actividad comercial del usuario
  - `sales` → productos vendidos

- Source: interna
- Granularity: mensual
- Period: 2018–2019
- Dataset: 
  - `customer_products` → productos del usuario
  - `customer_sociodemographics` → informacion del usuario
  - `product_description` → descripcion del producto

---

## Methodology
1. **Exploratory Data Analysis (EDA)**
   - Visualizar tendencias , analizar atributos , especificar el target.
   - imputacion de nulos y anomalias.
2. **Baseline Models**
   - Modelos basico de clasificaciónRegresion logistica
   - Evaluar accurracy , recall , Precision , F1 y ROC AUC 
3. **Classification models**
   - Evaluar overfiting y el mejor metrica de evaluación
   - Evaluar parametros con GridSearchCV
   - Train/test dividido en 80/20 
4. **Evaluation**
   - Compare baseline vs diferentes modelos
   - Visualize predictions vs actual sales

---

## Results
| Model | F1 | ROC_AUC |
|-------|-----|------|
| LogisticRegression (Baseline) | 0.5160 | 0.8770 |
| DecisionTreeClassifier | 0.7320 | 0.9340 |
| XGBClassifier | **0.7450** | **0.9480** |
| RandomForestClassifier | 0.6790 | 0.9410 |


---

## Business Impact
Tu modelo predice qué clientes tienen mayor probabilidad de abrir una cuenta o tarjeta. Esto tiene impacto directo en:

🎯 a) Incremento de conversiones
 - Puedes enfocar campañas solo en clientes con alta probabilidad
 - Evitas gastar en clientes que no van a convertir

 💰 b) Reducción de costes de marketing
 - Menos llamadas, emails o campañas innecesarias
 - Mayor ROI en campañas de captación

---

## Limitaciones

❌ 1. No es perfecto (errores de predicción)
 - Falsos positivos → contactas clientes que no comprarán
 - Falsos negativos → pierdes clientes que sí comprarían

 ❌ 2. Dependencia de los datos
 - Si los datos están sesgados → el modelo también
 - Si cambian los hábitos del cliente → el modelo se degrada


---

## Next Steps

1. Ajustar el threshold (MUY importante)
2. Interpretabilidad (clave en banca)
 Usa:
  - SHAP values
  - feature importance

---

## How to Run

### 1️⃣ Clone the repo
```bash
git clone https://github.com/Angel789data/EasyMoney_Capstone.git
