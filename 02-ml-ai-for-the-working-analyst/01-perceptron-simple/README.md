# Semana 1 - Perceptron Simple

## Clase

[NotebookLM - El Perceptron y la Geometria del Aprendizaje](https://notebooklm.google.com/notebook/9e83a39c-b17d-498f-beaf-1a36dcdf224f)

## Notas

### Ecosistema de Librerias y Herramientas

| Libreria | Funcion en el Proyecto |
| :--- | :--- |
| **numpy** | Maneja las imagenes como matrices numericas y permite operaciones matematicas rapidas |
| **pandas** | Gestiona la base de datos (CSV) con las etiquetas de "sano" (0) o "cancer" (1) |
| **skimage (io)** | Lee los archivos de imagen `.png` y los convierte en datos procesables |
| **sklearn** | Proporciona el modelo `Perceptron` y `train_test_split` para dividir los datos |
| **pickle** | Guarda el modelo entrenado (serializado) en un archivo para usarlo despues |
| **matplotlib / seaborn** | Visualizacion de datos y graficas de rendimiento |

### Metodologia del Proyecto

1. **Carga de Etiquetas**: Lectura de archivos CSV con nombres de archivo y su clase (0 = sano, 1 = canceroso)
2. **Preparacion de Imagenes**: Descompresion de archivos `.zip` y carga de imagenes desde almacenamiento
3. **Division de Datos**: Separacion de variables independientes ($X$, imagenes) y dependiente ($y$, etiquetas)
4. **Entrenamiento**: Ajuste del modelo Perceptron con datos de entrenamiento
5. **Evaluacion y Persistencia**: Medicion del desempeno y respaldo del modelo con pickle

### La Imagen como Datos

- Imagenes de $260 \times 260$ pixeles leidas con `skimage.io.imread`
- Cada pixel tiene un valor de intensidad entre **0** (negro) y **255** (blanco)
- **Flattening (Aplanado)**: La matriz $260 \times 260$ se transforma en un vector de $67,600$ numeros para que el Perceptron pueda procesarlo
- Al aplanar se pierde la relacion espacial entre pixeles vecinos verticalmente

### Arquitectura del Perceptron

El Perceptron es la red neuronal mas simple: una sola neurona que realiza clasificacion binaria.

**Componentes:**

- **Pesos** ($w$): Importancia asignada a cada pixel. Actuan como "controles de volumen"
- **Sesgo** ($b$): Termino de ajuste que permite decidir incluso con entradas minimas
- **Funcion de activacion**: Funcion escalon que decide si la neurona "dispara" o no

**Suma ponderada:**

$$z = \vec{w} \cdot \vec{x} + b$$

**Funcion de activacion (escalon):**

$$y = \begin{cases} 1 & \text{si } z > 0 \\ 0 & \text{si } z \leq 0 \end{cases}$$

### Regla de Aprendizaje del Perceptron

1. **Prediccion**: Multiplica cada pixel por su peso, suma todo y decide
2. **Calculo del error**: Compara la prediccion con la etiqueta real
3. **Actualizacion de pesos**:

$$w_{nuevo} = w_{actual} + \eta \cdot (y_{real} - y_{predicho}) \cdot x$$

Donde $\eta$ (`eta0`) es la **tasa de aprendizaje** que controla que tan brusco es el cambio.

### Hiperparametros

```python
modelo = Perceptron(max_iter=1000, eta0=0.1, random_state=42)
```

- `eta0` muy grande (ej. 10.0): Cambios drasticos, riesgo de no converger
- `eta0` muy pequeno (ej. 0.0001): Aprendizaje lento, posible estancamiento
- Tecnicas avanzadas: **Grid Search** y **Learning Rate Schedulers**

### Conceptos Clave

- **Clasificacion Binaria**: La salida solo puede ser 0 (sano) o 1 (cancer)
- **Desbalance de Clases**: 1,469 imagenes sanas vs 3,594 cancerosas — el modelo puede sesgarse
- **Algoritmo vs Modelo**: El algoritmo (Perceptron de sklearn) es estatico; el modelo (archivo `.sav`) es dinamico y depende de los datos
- **Funcion Sigmoide**: Alternativa al escalon que da probabilidades (0 a 1), lo que convierte al modelo en **Regresion Logistica**
- **Hiperplano**: Superficie de decision en un espacio de muchas dimensiones que divide lo sano de lo maligno
- **Maldicion de la dimensionalidad**: A mayor tamanio del vector, mas dificil y lento es encontrar los pesos optimos

### Visualizacion del Modelo

1. **Desplanar (Reshape)**: Convertir el vector de pesos de vuelta a $260 \times 260$ para crear un mapa de calor de lo que "mira" el modelo
2. **Reduccion de dimensionalidad (PCA / t-SNE / UMAP)**: Comprimir las 67,600 dimensiones a 2D para visualizar si las clases son separables

### Validacion y Metricas Medicas

- Uso de `train_test_split` para evaluar con datos que el modelo nunca vio
- En diagnostico medico, la **Sensibilidad (Recall)** es mas importante que la Precision pura — un falso negativo (no detectar cancer) es muy peligroso

## Codigo

_Notebooks y scripts._

## Recursos

_Links y materiales._
