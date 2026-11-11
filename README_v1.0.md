# A Log-Sine Analogue of Euler’s Constant: The Decay Rate of a Sine Product at Primes

- LogSine_Constant_v1.0.pdf es el artículo. LogSine_Constant_v1.0.tex es
  el original en LaTeX y LogSine_Constant_v1.0.txt es su copia idéntica.
- OEIS_submission_v1.0.txt contiene el borrador de la ficha para OEIS
  y enlaza al PDF público de esta versión. Si el autor añade un
  comentario, puede partir del significado del límite primo o de su
  relación con la distribución de partes fraccionarias en las secciones
  1 y 3, siguiendo las [normas de OEIS](https://oeis.org/wiki/Use_of_AI_for_OEIS_Submissions).
- certificate_Cg_v1.0.txt registra el cálculo por intervalos de los
  primeros 100 decimales de C_g. verify_Cg_v1.0.py permite reproducirlo
  con Python y mpmath 1.4.1. Los términos de Data se indexan desde 0.
- castor_v5/ reúne las doce figuras del artículo, sus scripts y datos de
  entrada, y los certificados históricos. El commit
  [0092616 de PaperMath](https://github.com/vcasana1/PaperMath/commit/0092616e38da579f3365cafba77b450b6a3f5fd2)
  documenta los antecedentes y contiene las versiones originales de
  las gráficas. Las cinco superficies y el mapa de ceros de esta entrega
  se regeneraron con el script figures_v4.py de este repositorio. En las
  superficies, cada marcador usa el módulo calculado en su vértice.
  El mapa de ceros usa un fondo claro. figures_v4_output.txt registra
  los residuos y la altura máxima de los marcadores de cada superficie.
  El README de cut_G_A_zeta/ describe el HTML y los archivos del corte.

Desde la raíz del paquete, compila con latexmk -pdf LogSine_Constant_v1.0.tex.
También puedes ejecutar dos veces pdflatex LogSine_Constant_v1.0.tex.
La compilación usa las figuras incluidas.

Para reproducir las figuras, ejecuta los siguientes scripts desde una copia
de castor_v5/, en el orden indicado. Requieren NumPy, SciPy, mpmath y
Matplotlib. figures_v4.py puede usar python-flint para acelerar la evaluación
de zeta.

1. figures_kernel_real.py
2. figure_gram.py
3. figures_v4.py
4. expsum_v3.py

Los registros de los certificados históricos identifican las versiones de
Arb y python-flint. Comprueba la integridad del paquete con
shasum -a 256 -c SHA256SUMS_v1.0.txt desde la raíz.
