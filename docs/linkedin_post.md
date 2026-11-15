# Borrador para LinkedIn

Dos versiones. Están escritas para publicarse cuando el repositorio sea público y el panel lleve unas semanas funcionando: así el enlace enseña un historial real, no una promesa.

Sugerencia de imagen: captura del panel (la gráfica "100 € invertidos en enero de 2017" con la zona del examen a ciegas sombreada), o un vídeo corto recorriéndolo.

---

## Español

Hace unas semanas tenía un bot que operaba cripto con una IA decidiendo cada entrada. Hizo dos operaciones. Las dos saltaron por stop. Una duró 19 minutos.

En vez de darle otro prompt, quise entender por qué. La respuesta no estaba en el mercado, estaba en la aritmética:

• Pagaba un 0,80 % por operación para buscar movimientos del 3 %.
• Ponía el stop a 1,6 % en un activo que se mueve un 2 % al día. Probabilidad de que saltara por puro ruido en 24 horas: 44 %.
• El coste de la IA equivalía a un 29 % anual del capital.

Así que lo reconstruí como lo haría un equipo cuantitativo, con reglas fijas y sin opiniones:

→ Filtros de Kalman para estimar la tendencia con su margen de error
→ Un modelo de Markov oculto para detectar regímenes de mercado
→ Teoría de matrices aleatorias para separar correlación real de ruido
→ Paridad de riesgo jerárquica entre acciones, oro, plata y cripto

Y lo puse a prueba en serio: lo diseñé solo con datos hasta 2021 y guardé 2022–2026 bajo llave para examinarlo una única vez.

Resultado del examen a ciegas: +11,9 % anual, con una caída máxima del −11 %. El S&P 500 dio lo mismo con una caída del −23 %. Bitcoin, −65 %.

Lo que más aprendí fue lo que NO funcionó:

1. El seguimiento de tendencia en acciones no cubre sus comisiones. Lo quité.
2. Las piezas sofisticadas apenas añaden rentabilidad. Diversificar por riesgo y operar poco pesa más.
3. No hay alfa: el modelo no adivina el mercado, controla el riesgo. Una cartera simple 60/40 de bolsa y oro ganó más esos años.
4. De 177 acciones tokenizadas listadas en mi bróker, muchas no se negocian. Su precio quieto parecía "bajo riesgo" y el modelo quería cargar de ellas.

Pedí a otra IA que auditara el código buscando trampas antes de mirar el resultado final. Encontró 16 problemas. Todos están corregidos o documentados, uno por uno.

Todo es público: código, metodología, errores y un panel que se actualiza solo cada noche.

Panel: https://juanmariarodriguez2-png.github.io/hermes-quant/
Código: https://github.com/juanmariarodriguez2-png/hermes-quant

Estudio Economía y Data Science en CUNEF y busco prácticas de verano en mercados, riesgos o datos. Si trabajas en esto y ves algo mejorable, me encantaría leerte.

#QuantitativeFinance #DataScience #Python #RiskManagement

---

## English

A few weeks ago I had a bot trading crypto with an AI deciding every entry. It made two trades. Both were stopped out. One lasted 19 minutes.

Instead of writing it a better prompt, I tried to understand why. The answer was not in the market. It was in the arithmetic:

• It paid 0.80% per trade to chase 3% moves.
• It placed a stop 1.6% away on an asset that moves 2% a day. Chance of being stopped out by noise alone within 24 hours: 44%.
• The AI calls cost the equivalent of 29% of the capital per year.

So I rebuilt it the way a quant team would, with fixed rules and no opinions:

→ Kalman filters to estimate trend with an error bar
→ A hidden Markov model for market regimes
→ Random matrix theory to separate real correlation from noise
→ Hierarchical risk parity across equities, gold, silver and crypto

Then I tested it properly: designed on data up to 2021 only, with 2022–2026 locked away to be examined once.

Blind-exam result: +11.9% a year with a worst drawdown of −11%. The S&P 500 returned the same with a −23% drawdown. Bitcoin, −65%.

What I learned most from is what did NOT work:

1. Trend-following on equities does not cover its commissions. I removed it.
2. The sophisticated parts barely add return. Diversifying by risk and trading rarely matters more.
3. There is no alpha: the model does not outguess the market, it controls risk. A plain 60/40 stocks-gold mix returned more over those years.
4. Of 177 tokenized stocks listed at my broker, many do not trade. Their frozen prices looked "low risk" and the model wanted to load up on them.

I asked another AI to audit the code for flaws before looking at the final result. It found 16 problems. Every one is fixed or documented, one by one.

Everything is public: code, method, mistakes, and a dashboard that updates itself every night.

Dashboard: https://juanmariarodriguez2-png.github.io/hermes-quant/
Code: https://github.com/juanmariarodriguez2-png/hermes-quant

I study Economics and Data Science at CUNEF and I am looking for a summer internship in markets, risk or data. If you work in this field and see something to improve, I would love to hear it.

#QuantitativeFinance #DataScience #Python #RiskManagement
