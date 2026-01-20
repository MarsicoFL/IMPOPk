# Análisis Crítico del Modelo de Detección de IBD

**Fecha**: 2026-01-16
**Revisores**: Experto en Genética de Parentesco + Experto en Matemáticas/Estadística

---

## Resumen Ejecutivo

El modelo HMM implementado para detección de IBD presenta **debilidades teóricas significativas** que cuestionan la validez de los resultados obtenidos. Las principales preocupaciones son:

1. **Parámetros de emisión mal calibrados** (σ subestimado ~3x)
2. **Sesgo de truncamiento** en los datos de entrada
3. **Separabilidad insuficiente** entre distribuciones IBD/non-IBD (d' < 1)
4. **Discrepancia masiva** entre %IBD esperado vs observado

---

## 1. Perspectiva del Genetista de Parentesco

### 1.1 Fundamento Biológico del Modelo

El modelo asume dos estados ocultos:
- **IBD (Identity-by-Descent)**: Segmentos heredados de un ancestro común
- **non-IBD**: Similitud por convergencia evolutiva (IBS but not IBD)

**Crítica**: El modelo trata IBD como un estado binario, pero en realidad:
- IBD puede ser IBD1 (1 alelo) o IBD2 (2 alelos) en diploides
- La "profundidad" del IBD (generaciones al ancestro común) afecta la longitud esperada
- El modelo no distingue entre IBD reciente (largo) e IBD ancestral (corto, fragmentado)

### 1.2 Parámetros de Diversidad Poblacional

| Población | π (modelo) | Fuente |
|-----------|------------|--------|
| AFR | 0.00125 | 1000 Genomes |
| EUR | 0.00085 | 1000 Genomes |
| EAS | 0.00080 | 1000 Genomes |

**Validación**: Los valores de π son razonables y consistentes con literatura (Auton et al. 2015).

**Problema**: El modelo usa π para derivar la media de identidad non-IBD como `1 - π`, pero esto solo es válido para:
- Sitios **bialélicos independientes**
- Sin considerar **LD (linkage disequilibrium)**
- Sin considerar **errores de alineamiento** específicos del pangenoma

### 1.3 Resultados vs Expectativas Biológicas

| Resultado | Valor Observado | Expectativa Biológica | Evaluación |
|-----------|-----------------|----------------------|------------|
| IBD más largo (EUR) | 3.76 Mb | Parientes 2°-3° grado | ⚠️ Puede ser artefacto |
| IBD más largo (AFR) | 3.27 Mb | Inusual sin parentesco conocido | ⚠️ Sospechoso |
| IBD más largo (EAS) | 4.26 Mb | Muy largo para no-relacionados | ⚠️ Requiere verificación |
| Mean segment EUR | 251 kb | Típico para IBD poblacional | ✓ Razonable |
| Fraction IBD (EAS) | 34% | Muy alto para individuos aleatorios | ❌ Probablemente sobredetección |

**Interpretación del Genetista**: Los segmentos IBD de 3-4 Mb en individuos "no relacionados" de HPRC son altamente sospechosos. Esperaríamos:
- ~20 segmentos IBD > 1 Mb entre primos segundos
- ~0-2 segmentos > 1 Mb entre individuos aleatorios de la misma población

La detección de cientos de segmentos largos sugiere **sobredetección sistemática**.

---

## 2. Perspectiva Matemática/Estadística

### 2.1 Especificación del HMM

El modelo es un HMM de 2 estados con:
- **Emisiones**: Gaussianas `N(μ, σ²)`
- **Transiciones**: Matriz estacionaria 2×2

**Formulación correcta**: ✓ La implementación del forward-backward y Viterbi es matemáticamente correcta.

### 2.2 Calibración de Distribuciones de Emisión

#### Problema 1: Varianza Subestimada

| Parámetro | Modelo | Empírico | Ratio |
|-----------|--------|----------|-------|
| σ_non-IBD (AFR) | 0.00087 | 0.00218 | **2.5x** |
| σ_non-IBD (EUR) | 0.00071 | 0.00207 | **2.9x** |
| σ_non-IBD (EAS) | 0.00069 | 0.00210 | **3.0x** |

La fórmula usada:
```
σ_non_ibd = sqrt(π / window_size * ld_correction)
         = sqrt(0.001 / 5000 * 3)
         ≈ 0.0007
```

**Problema**: El factor `ld_correction = 3` es arbitrario y claramente **insuficiente**. La varianza empírica sugiere que el factor debería ser ~20-25.

#### Problema 2: Separabilidad (d-prime)

La métrica d' mide la separación entre distribuciones:

```
d' = |μ_IBD - μ_non-IBD| / sqrt((σ²_IBD + σ²_non-IBD)/2)
```

| Población | d' (modelo) | d' (empírico) | Interpretación |
|-----------|-------------|---------------|----------------|
| AFR | 1.34 | 0.65 | Pobre |
| EUR | 0.90 | 0.38 | Muy pobre |
| EAS | 0.83 | 0.34 | Muy pobre |

**Regla general**:
- d' > 3: Excelente discriminación
- d' > 2: Buena discriminación
- d' < 2: Alta tasa de error
- d' < 1: Discriminación casi aleatoria

**Conclusión matemática**: Con d' ≈ 0.3-0.7, las distribuciones se solapan extensamente. La clasificación basada solo en emisiones sería casi aleatoria.

### 2.3 Rol del HMM en la Detección

El HMM "salva" parcialmente el problema del solapamiento mediante:
1. **Información temporal**: Estados consecutivos tienden a ser iguales
2. **Prior de transición**: P(enter IBD) = 0.0001 penaliza fuertemente entrar en IBD

**Matemáticamente**: Sea L(x) = log P(emit_IBD(x)) - log P(emit_non-IBD(x)). Para clasificar un punto como IBD se requiere:

```
L(x) > log(P(non-IBD→non-IBD) / P(non-IBD→IBD))
     = log((1-0.0001) / 0.0001)
     ≈ log(10000)
     = 9.2
```

Esto significa que la **razón de verosimilitudes debe ser > 10,000:1** para "entrar" en IBD desde non-IBD. Pero con el solapamiento observado (d' < 1), esto solo ocurre en la cola extrema.

**Paradoja**: El modelo detecta IBD principalmente por **continuidad temporal**, no por el valor de identidad per se. Esto explica por qué detecta segmentos largos pero puede estar capturando runs de alta identidad IBS que no son verdaderamente IBD.

### 2.4 Sesgo de Selección en los Datos

Los datos de entrada (`*_ibs.tsv`) están **pre-filtrados** por cutoff ≥ 0.99.

**Consecuencia matemática**:
- La distribución non-IBD observada es la **distribución truncada** `f(x | x > 0.99)`
- Esto sesga la media hacia arriba y reduce la varianza aparente
- El modelo asume distribuciones completas, no truncadas

**Efecto**: El modelo "ve" una distribución non-IBD que parece más similar a IBD de lo que realmente es.

### 2.5 Distribución Estacionaria vs Resultados

La distribución estacionaria del HMM es:
```
π_IBD = P(enter) / (P(enter) + P(exit))
      = 0.0001 / (0.0001 + 0.01)
      = 0.0099 ≈ 1%
```

**Observado**: 11-34% IBD (10-34x mayor)

**Explicaciones posibles**:
1. **Sesgo de selección**: Solo analizamos pares con ≥50 windows IBD
2. **Sobredetección**: Runs de alta identidad IBS clasificados como IBD
3. **Parentesco no documentado**: Algunos individuos de HPRC podrían ser parientes lejanos

---

## 3. Recomendaciones

### 3.1 Mejoras Inmediatas

1. **Recalibrar σ_non_ibd**: Usar varianza empírica o aumentar `ld_correction` a ~25
2. **Modelar distribución truncada**: Ajustar emisiones para datos filtrados
3. **Validar con datos sintéticos**: Generar ground truth conocido

### 3.2 Mejoras de Modelo

1. **Distribuciones de emisión mixtas**: Usar mezcla de Gaussianas para capturar heterogeneidad
2. **β-distribution**: Identidad está en [0,1], una Beta puede ser más apropiada
3. **Incorporar información de LD**: Windows cercanas no son independientes

### 3.3 Validación Necesaria

1. **Comparar con IBD conocido**: Usar tríos/familias de HPRC para validar
2. **Benchmark vs otras herramientas**: Comparar con GERMLINE, IBDseq, hap-IBD
3. **Análisis de sensibilidad**: Variar parámetros y evaluar estabilidad

---

## 4. Conclusión

El modelo actual tiene una **base teórica sólida** (HMM con forward-backward) pero sufre de **calibración deficiente** de los parámetros de emisión. Los resultados muestran patrones biológicamente plausibles en algunos aspectos (diferencias poblacionales, longitudes de segmento) pero las fracciones IBD detectadas son sospechosamente altas.

**Recomendación**: Los resultados deben interpretarse como **indicativos** más que **definitivos**. Se requiere validación adicional antes de publicación.

---

## Apéndice: Fórmulas Clave

### A1. Diversidad nucleotídica y identidad esperada
```
E[identidad | non-IBD] = 1 - π
Var[identidad | non-IBD] ≈ π(1-π)/L × C_LD
```
donde L = tamaño de ventana, C_LD = factor de corrección por LD.

### A2. Forward-backward
```
P(z_t = k | x_{1:T}) = α_t(k) × β_t(k) / P(x_{1:T})
```

### A3. Separabilidad d'
```
d' = |μ_1 - μ_2| / sqrt((σ_1² + σ_2²)/2)
```

### A4. Distribución estacionaria
```
π = [P(exit)/(P(enter)+P(exit)), P(enter)/(P(enter)+P(exit))]
```
