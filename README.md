# 🧠 Redes Neuronales con TensorFlow

> Colección de prácticas y fundamentos teóricos sobre redes neuronales implementadas en Python utilizando la biblioteca TensorFlow.

Este repositorio sirve como guía de estudio y referencia para entender la estructura, funcionamiento y métodos de aprendizaje de las arquitecturas de redes neuronales más comunes.

---

## 📑 Tabla de Contenidos
1. [Clasificación de Redes Neuronales](#clasificación-de-redes-neuronales-arquitecturas)
2. [Tipos de Aprendizaje](#tipos-de-aprendizaje)
3. [Fundamentos Matemáticos](#-fundamentos-matemáticos)

---

## 🏗️ Clasificación de Redes Neuronales (Arquitecturas)

A continuación se detallan los tipos de redes neuronales fundamentales y su mecanismo de funcionamiento.

### 1. Red Neuronal Monocapa (Perceptrón Simple)

Es la forma más básica de red neuronal. Consta de una única capa de neuronas de salida conectadas directamente a las entradas a través de pesos sinápticos. Solo puede resolver problemas que son linealmente separables.

**¿Cómo funciona?**
1.  Recibe un conjunto de valores de entrada.
2.  Cada entrada se multiplica por un peso asociado y se suman todos los resultados (suma ponderada). Se añade un término de sesgo (bias).
3.  El resultado pasa por una función de activación (generalmente tipo escalón).
4.  Si la suma supera un umbral, la neurona se activa (salida 1), si no, permanece inactiva (salida 0).

<div align="center">
  <img src="https://wsrv.nl/?url=upload.wikimedia.org/wikipedia/commons/thumb/6/60/ArtificialNeuronModel_english.png/640px-ArtificialNeuronModel_english.png&bg=white&w=500" width="500" alt="Esquema Perceptrón Simple">
  <br>
  <em>Modelo matemático de un Perceptrón Simple</em>
</div>

### 2. Red Neuronal Multicapa (Perceptrón Multicapa - MLP)

Es una evolución del perceptrón simple que incluye una o más "capas ocultas" entre la entrada y la salida. Es capaz de resolver problemas no lineales complejos gracias a esta profundidad.

**¿Cómo funciona?**
1.  La información fluye hacia adelante (feedforward) desde la capa de entrada, pasando por las capas ocultas, hasta la salida.
2.  Cada neurona en una capa oculta procesa la información de la capa anterior usando funciones de activación no lineales (como Sigmoid, Tanh o ReLU).
3.  El aprendizaje se realiza mediante el algoritmo de **Backpropagation** (propagación hacia atrás), que calcula el error en la salida y ajusta los pesos de todas las capas anteriores para minimizar ese error.

<div align="center">
  <img src="https://wsrv.nl/?url=upload.wikimedia.org/wikipedia/commons/thumb/4/46/Colored_neural_network.svg/500px-Colored_neural_network.svg.png&bg=white&w=400" width="400" alt="Esquema Perceptrón Multicapa">
  <br>
  <em>Arquitectura multicapa con entrada, capas ocultas y salida</em>
</div>

### 3. Red Neuronal Convolucional (CNN)

Diseñadas específicamente para procesar datos con una estructura tipo rejilla, como las imágenes. Son excelentes para detectar patrones espaciales (bordes, formas, texturas).

**¿Cómo funciona?**
1.  Utiliza "filtros" o "kernels" que se deslizan (convolución) sobre la imagen de entrada para extraer mapas de características.
2.  Las primeras capas detectan características simples (líneas), mientras que las capas más profundas detectan formas complejas (ojos, caras).
3.  Suelen intercalar capas de *Pooling* (agrupamiento) para reducir la dimensionalidad y hacer que la red sea más robusta a variaciones de posición.

<div align="center">
  <img src="https://wsrv.nl/?url=upload.wikimedia.org/wikipedia/commons/6/63/Typical_cnn.png&bg=white&w=700" width="700" alt="Esquema CNN">
  <br>
  <em>Flujo típico de una CNN: Convolución -> Pooling -> Fully Connected</em>
</div>

### 4. Red Neuronal Recurrente (RNN)

Están diseñadas para trabajar con datos secuenciales donde el orden importa (series temporales, texto, audio). A diferencia de las anteriores, tienen "memoria".

**¿Cómo funciona?**
1.  Procesan la información paso a paso.
2.  La salida de una neurona en el paso de tiempo `t-1` se utiliza como parte de la entrada para la misma neurona en el paso de tiempo `t`.
3.  Esto forma un bucle que permite a la red "recordar" información de estados previos e influir en las predicciones futuras basándose en el contexto histórico.

<div align="center">
  <img src="https://wsrv.nl/?url=upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Recurrent_neural_network_unfold.svg/640px-Recurrent_neural_network_unfold.svg.png&bg=white&w=600" width="600" alt="Esquema RNN">
  <br>
  <em>RNN "desenrollada" en el tiempo mostrando la memoria secuencial</em>
</div>

### 5. Red de Base Radial (RBF)

Son redes generalmente de tres capas (entrada, oculta, salida) que utilizan funciones de base radial como funciones de activación en la capa oculta. Son muy buenas para aproximación de funciones e interpolación.

**¿Cómo funciona?**
1.  La capa de entrada solo distribuye los datos.
2.  Las neuronas de la capa oculta calculan la "distancia" (generalmente euclidiana) entre el vector de entrada y un centro prototipo almacenado en la neurona.
3.  La activación depende de qué tan cerca esté la entrada de ese centro. Cuanto más cerca, mayor es la activación (siguiendo una forma de campana de Gauss).
4.  La capa de salida realiza una combinación lineal de las activaciones de la capa oculta.

<div align="center">
  <img src="https://wsrv.nl/?url=upload.wikimedia.org/wikipedia/commons/thumb/7/7d/Radial_funktion_network.svg/500px-Radial_funktion_network.svg.png&bg=white&w=450" width="450" alt="Esquema RBF">
  <br>
  <em>Red RBF mostrando las funciones radiales en la capa oculta</em>
</div>

---

## 🎓 Tipos de Aprendizaje

Clasificación de las redes según el método utilizado para ajustar sus pesos durante el entrenamiento.

### 1. Aprendizaje Supervisado

> Se caracteriza porque el proceso de aprendizaje se realiza mediante un entrenamiento controlado por un "supervisor" (tenemos los datos de entrada y sus etiquetas de salida correctas).

El objetivo es que la red aprenda la relación entre la entrada y la salida deseada. Si la salida de la red no es correcta, se modifican los pesos de las conexiones para que la salida obtenida se aproxime a la deseada.

A su vez, se puede subdividir en:

* **Aprendizaje por corrección de error:**
    Ajusta los pesos de las conexiones en función de la diferencia entre los valores esperados y los obtenidos.
    * *Ejemplos de algoritmos:*
        * Perceptrón
        * Delta o Mínimo error cuadrado (LMS Error)
        * Backpropagation (LMS multicapa)

* **Aprendizaje estocástico:**
    Realiza cambios aleatorios sobre los pesos y evalúa si la predicción mejora o empeora. Se conservan los cambios que mejoran los resultados.

### 2. Aprendizaje No Supervisado (o Autosupervisado)

> Se caracteriza porque no requiere influencia externa ni etiquetas para ajustar los pesos. La red trabaja solo con los datos de entrada.

Este tipo de aprendizaje busca encontrar estructuras ocultas, patrones, correlaciones o categorías dentro de los datos sin etiquetar. La salida suele representar el grado de similitud entre datos o un agrupamiento (clustering).

A su vez, se puede subdividir en:

* **Aprendizaje Hebbiano:**
    Se basa en la regla de que "las neuronas que se activan juntas, permanecen conectadas". Permite medir la familiaridad o extraer características fuertes de los datos de entrada reforzando conexiones utilizadas frecuentemente.

* **Aprendizaje Competitivo y Comparativo:**
    Las neuronas compiten entre sí para responder a un subconjunto de datos de entrada. Permite realizar clasificaciones: se añaden elementos a una clase existente si se determina similitud, o se crea una nueva clase con pesos propios si el elemento es muy diferente.

### 3. Aprendizaje por Refuerzo

> Es un aprendizaje basado en la interacción con un entorno. La red (agente) no recibe la salida exacta, sino una señal de "recompensa" o "castigo" indicando si su acción fue buena o mala.

Se considera un aprendizaje más lento que el de corrección de errores. El algoritmo ajusta los pesos basándose en mecanismos de probabilidad para maximizar la recompensa acumulada a largo plazo. Es típico en robótica y juegos.

---

## 🧮 Fundamentos Matemáticos

Las redes neuronales no son "cajas mágicas", sino estructuras basadas en álgebra lineal y cálculo. Estas son las fórmulas clave que gobiernan su funcionamiento.

### 1. La Neurona Artificial (Suma Ponderada)
Antes de activar la neurona, calculamos la suma ponderada de las entradas más el sesgo (*bias*).

$$z = \sum_{i=1}^{n} (x_i \cdot w_i) + b$$

Donde:
* $x$: Vector de entrada.
* $w$: Vector de pesos (weights).
* $b$: Sesgo (bias), que permite desplazar la función de activación.

### 2. Funciones de Activación
Deciden si una neurona se activa o no y añaden no-linealidad al sistema.

* **Sigmoide (Sigmoid):** Transforma valores a un rango (0, 1). Ideal para probabilidades.
    $$\sigma(z) = \frac{1}{1 + e^{-z}}$$

* **ReLU (Rectified Linear Unit):** La más usada en capas ocultas. Si es positivo se mantiene, si es negativo se vuelve cero.
    $$f(z) = \max(0, z)$$

* **Softmax:** Usada en la capa de salida para clasificación multiclase. Convierte el vector de salidas en una distribución de probabilidad.
    $$\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

### 3. Función de Coste (Loss Function)
Mide el error de la red, es decir, qué tan lejos está la predicción ($\hat{y}$) del valor real ($y$).

* **Error Cuadrático Medio (MSE):** Usado para regresión.
    $$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

* **Entropía Cruzada (Cross-Entropy):** Usada para clasificación.
    $$Loss = - \sum_{i=1}^{C} y_i \cdot \log(\hat{y}_i)$$

### 4. Descenso del Gradiente (Actualización de Pesos)
Para que la red aprenda, debemos minimizar el error ajustando los pesos. Usamos la derivada del error respecto al peso.

$$w_{nuevo} = w_{actual} - \alpha \cdot \frac{\partial Loss}{\partial w}$$

Donde:
* $\alpha$ (alfa): **Tasa de aprendizaje (Learning Rate)**. Controla qué tan rápido o lento aprende la red.
* $\frac{\partial Loss}{\partial w}$: El gradiente, que indica la dirección en la que debemos mover los pesos para reducir el error.
