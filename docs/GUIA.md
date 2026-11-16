# Guía de puesta en marcha

Todo lo necesario para dejar Hermes Quant funcionando solo, cada noche, sin depender de ningún ordenador.

## 1. Lo que hay y lo que hace

Cada noche, hacia las 01:23 UTC (las 2:23 en España en invierno, las 3:23 en verano), GitHub ejecuta el modelo:

1. Descarga las velas del día de Kraken (12 criptomonedas, 8 fondos tokenizados, el euro/dólar y las acciones de tu lista de favoritas).
2. Recalcula volatilidades, tendencias, régimen de mercado y correlaciones.
3. Una vez por semana fija los pesos objetivo; el resto de días solo opera si algún peso se ha desviado lo bastante como para compensar la comisión.
4. Ejecuta las órdenes (simuladas por defecto), guarda el estado en el propio repositorio y actualiza el panel.

No hay servidor, ni Docker, ni coste de IA. Si el portátil está apagado, da igual.

## Dónde verlo

- **Panel:** https://juanmariarodriguez2-png.github.io/hermes-quant/ — el informe: qué ve el modelo, qué tiene la cuenta y la evidencia que lo respalda. Cambia una vez al día, tras la ejecución nocturna.
- **Terminal en vivo:** https://juanmariarodriguez2-png.github.io/hermes-quant/terminal.html — la pantalla de seguimiento: precios, velas, libro de órdenes y operaciones del mercado en tiempo real (canal público de Kraken), junto a la señal del modelo para cada mercado, el valor de la cuenta al momento, las órdenes pendientes y el registro de lo que decidió el bot en su última ejecución. Escribe un símbolo y pulsa Intro para cambiar de mercado.

El modelo decide una vez al día y opera poco, así que en el terminal lo que se mueve es el mercado; las decisiones cambian cada noche.

## 2. Publicarlo (una sola vez, 3 minutos)

1. Sube la carpeta a un repositorio de GitHub llamado `hermes-quant`.
2. En el repositorio: **Settings → Pages → Build and deployment → Source: GitHub Actions.**
3. **Actions → Hermes daily run → Run workflow** para lanzar la primera ejecución a mano.

A los dos minutos el panel estará en https://juanmariarodriguez2-png.github.io/hermes-quant/.

No hace falta tocar los permisos de Actions: el flujo pide por sí mismo los que necesita.

GitHub Pages es gratis en repositorios públicos. Si el repositorio es privado, o mientras no actives el paso 2, el modelo funciona igual y guarda su estado cada noche; simplemente no publica el panel, y lo dice en el registro de la ejecución.

## 3. Avisos por Telegram (opcional)

1. En Telegram, habla con `@BotFather`, escribe `/newbot` y guarda el token.
2. Escribe cualquier cosa a tu bot nuevo y abre `https://api.telegram.org/botTOKEN/getUpdates` para ver tu `chat.id`.
3. En GitHub: **Settings → Secrets and variables → Actions → New repository secret**. Crea `TELEGRAM_BOT_TOKEN` y `TELEGRAM_CHAT_ID`.

Recibirás un mensaje diario con el valor de la cartera, las órdenes y las posiciones.

## 4. Pasar a dinero real

Hazlo solo después de ver el modo simulado funcionar unas semanas. Son tres pasos, y cada uno es un candado distinto.

**Candado 1: las claves.** En Kraken Pro: https://pro.kraken.com/app/settings/api (icono de persona → **Configuración → Conexiones y API → Crear clave API**). La página antigua de kraken.com ya es de solo lectura. Marca estos cinco permisos y ninguno más:

- Consulta (Query Funds)
- Consultar órdenes y operaciones abiertas (Query Open Orders & Trades)
- Consultar órdenes y operaciones cerradas (Query Closed Orders & Trades)
- Crear y modificar órdenes (Create & Modify Orders)
- Cancelar y cerrar órdenes (Cancel/Close Orders)

El de órdenes cerradas hace falta porque cada orden se ejecuta o se cancela al instante, y el modelo lee el resultado cuando ya está cerrada.

**Nunca marques "Retirar" ni "Agregar direcciones de retiro".** Sin esas casillas, nadie que robe la clave puede sacar dinero.

Deja las opciones adicionales como vienen. No restrinjas por dirección IP: GitHub ejecuta el modelo desde direcciones que cambian.

Kraken enseña la clave privada una sola vez. Cópiala directamente a GitHub, en **Settings → Secrets and variables → Actions → New repository secret**, como dos secretos: `KRAKEN_API_KEY` (la "API key") y `KRAKEN_API_SECRET` (la "Private key"). No la pegues en ningún otro sitio.

**Candado 2: el modo.** En **Settings → Secrets and variables → Actions → Variables** crea `HERMES_MODE` con valor `live`. A partir de aquí el modelo lee tu saldo real y **envía cada orden a Kraken solo para validarla**: Kraken comprueba que es correcta y la descarta. No se mueve ni un céntimo. Revisa el registro de la ejecución (pestaña Actions) durante varios días: debe mostrar órdenes con estado `validated` y sin errores.

**Candado 3: armarlo.** Cuando el registro te convenza, crea la variable `HERMES_LIVE_ARMED` con valor `yes`. Ahora sí opera.

Dos límites adicionales que puedes ajustar:

- `HERMES_MAX_ORDER_EUR` (por defecto 100): ninguna orden, conversiones de divisa incluidas, supera esa cantidad.
- `HERMES_PROFILE`: `balanced` (por defecto) o `growth`.

### Ensayo en seco, cuando quieras

En la pestaña **Actions** hay un flujo manual, **Kraken connection check**. Lee el saldo y las órdenes, y envía a Kraken las órdenes que el modelo colocaría hoy con la marca de validación: Kraken las comprueba y las descarta. No coloca nada. Su cliente se niega, por código, a enviar cualquier otra cosa. Úsalo antes de armar el modelo y después de cualquier cambio.

El 15 de noviembre de 2026 se ejecutó contra la cuenta real: claves, lectura de saldo, órdenes de criptomonedas y cambio de divisa, aceptadas; órdenes de acciones tokenizadas, rechazadas (ver el apartado siguiente). Todavía no se ha ejecutado ninguna orden real: lo validado es que Kraken las acepta, no cómo se ejecutan.

### Cuentas europeas: las acciones tokenizadas se colocan a mano

Kraken no admite órdenes de acciones tokenizadas (xStocks) por API ni en el libro de órdenes de Kraken Pro a clientes del Espacio Económico Europeo. Solo se pueden comprar y vender a mano en kraken.com, con euros, después de aprobar un cuestionario de idoneidad. Es una norma de Kraken, no un fallo de configuración.

Por eso, en modo real y por defecto:

- **Criptomonedas:** las opera el modelo solo, por API.
- **Bolsa, oro y plata (SPYx, QQQx, GLDx, SLVx):** el modelo no envía esas órdenes ni cambia euros a dólares para ellas. Las calcula cada noche y las deja como **órdenes pendientes de colocar a mano**: aparecen en un aviso en el panel, en el registro de la ejecución (líneas `BY HAND`) y en el mensaje de Telegram. Cuando las colocas en kraken.com, la ejecución siguiente lee tus saldos y el aviso desaparece.

Esta parte de la cartera se mueve poco (el modelo la mantiene siempre y solo la ajusta cuando un peso se desvía bastante), así que son cuatro compras al empezar y un ajuste de vez en cuando.

**Cómo se compran:** en Kraken Pro, botón **Convertir** (arriba a la derecha): de EUR al activo que acaba en «x» (GLDx, SPYx, QQQx, SLVx), el importe del aviso, **Revisar** y confirmar. El formulario de órdenes normal no sirve para estos activos en una cuenta europea.

**Cuánto cuesta:** Convertir muestra «comisión 0», pero el precio lleva un margen. Medido el 16 de noviembre de 2026 con una compra real de GLDx: 1,07 % sobre el precio de mercado en ese instante. Las pruebas del modelo suponían un 0,38 % (comisiones del libro de órdenes). Rehecho el examen a ciegas con el coste real, el resultado baja de 11,9 % a 10,9 % anual, con la misma caída máxima. Está en `research/08_execution_costs.py` y en la tabla de resultados del panel.

Si tu cuenta no es europea y Kraken sí te admite esas órdenes por API, crea la variable `HERMES_TOKENIZED_API` con valor `yes` y el modelo las operará también.

### Cómo se envían las órdenes

Todas son órdenes límite con un tope del 0,5 % sobre el precio de referencia y de tipo "inmediata o cancelada": o se ejecutan al momento a un precio aceptable, o no se ejecutan. No queda ninguna orden viva en el libro. Después el modelo lee de Kraken lo que de verdad se ha ejecutado; nunca lo da por hecho.

### Si ingresas o retiras dinero

No hay que hacer nada. En cada ejecución el modelo compara lo que la cuenta debería valer (lo que tenía ayer, a precios de hoy) con lo que vale. Si la diferencia es demasiado grande para ser comisiones (más de 5 € y del 3 %), la trata como dinero que ha entrado o salido: mueve con ella la base de la rentabilidad y el máximo de la cuenta, de modo que un ingreso no parece una ganancia ni una retirada una pérdida, y lo avisa en el panel. Lo mismo ocurre si compras con el dinero de la cuenta un activo que el modelo no gestiona: lo lista como «no reconocido» y no lo cuenta en el valor.

Si prefieres reiniciar del todo el contador de rentabilidad, crea la variable `HERMES_REBASE` con valor `yes`, deja pasar una ejecución y bórrala.

### El cortacircuitos

Si la cuenta cae un 25 % desde su máximo, el modelo vende todo, se queda en liquidez y avisa. No vuelve a operar hasta que alguien edite `data/live/state.json` y ponga `"halted": false`. Está pensado para que un fallo, del modelo o del mercado, no se convierta en una ruina.

## Pasar a Trading 212: que el bot compre solo, sin pasos a mano

En Kraken, una cuenta europea tiene que comprar a mano la bolsa, el oro y la plata, y cada compra cuesta alrededor de un 1 %. Trading 212 es un bróker regulado en la UE con acciones y fondos reales, sin comisión de compraventa, desde 1 €, y con una API por la que el bot envía todas las órdenes. El conector está hecho y probado contra un simulador; falta probarlo contra una cuenta de verdad, y eso empieza con dinero ficticio.

**Lo que haces tú (una vez)**

1. **Abre la cuenta** en trading212.com eligiendo **Invest** (no CFD) y verifica tu identidad.
2. **Crea una clave de práctica.** En la app o la web cambia a la cuenta de **práctica** (dinero ficticio) y entra en *Ajustes → API (Beta) → Generar clave API*. Marca todos los permisos y copia la clave y el secreto. Esa API no tiene ninguna función para retirar dinero.
3. **Guárdalos en GitHub**, nunca en un chat: *Settings → Secrets and variables → Actions → New repository secret*, uno llamado `T212_API_KEY` y otro `T212_API_SECRET`.
4. **Lanza el ensayo:** *Actions → Trading 212 connection check → Run workflow*. No envía nada. Dice si la clave funciona, con qué instrumento operaría cada mercado, si su bolsa está abierta y qué hay en la cuenta. También busca las acciones españolas y europeas y los fondos de Japón y China, y dice cuáles ofrece tu cuenta.
5. **Deja que opere con dinero ficticio:** crea la variable `HERMES_VENUE` con valor `trading212`. Desde ese momento el bot opera en la cuenta de práctica (la de Kraken se queda como está, sin tocar) y el panel lo muestra como «Trading 212 · ensayo».
6. **Dinero real, cuando el ensayo te convenza:** ingresa dinero en Trading 212, crea una clave en la cuenta real, sustituye los dos secretos y crea la variable `T212_ENV` con valor `live`.

**Cómo opera allí**

- Cada mercado del modelo se compra a través del fondo europeo que sigue lo mismo, porque una cuenta europea no puede comprar los fondos estadounidenses: S&P 500 → iShares Core S&P 500 UCITS; Nasdaq 100 → iShares Nasdaq 100 UCITS; oro → iShares Physical Gold; plata → iShares Physical Silver. Las acciones de tu lista son las acciones reales en su bolsa. Las criptomonedas, mediante productos cotizados (ETP) si tu cuenta los ofrece; si no, esa parte se queda en liquidez. El mapa está en `config/trading212.json`.
- Una acción solo se puede comprar con su bolsa abierta. Por eso hay dos ejecuciones más, de lunes a viernes a las 14:45 y 15:20 UTC (15:45 y 16:20 en España en invierno), cuando Europa y Nueva York están abiertas a la vez. La de madrugada sigue calculando; las órdenes salen por la tarde.
- Las señales siguen saliendo de los mismos datos de siempre; lo que cambia es dónde se compra.
- El dinero de Kraken no se mueve solo: para pasarlo hay que vender allí las cuatro posiciones, retirar los euros al banco e ingresarlos en Trading 212. Es el último paso a mano.

**Lo que conviene saber**

- La API de Trading 212 está en fase beta y puede cambiar. Para eso son el ensayo y la cuenta de práctica.
- No ofrece las bolsas de Japón ni de China directamente; sí fondos cotizados en Europa que las siguen.
- Para añadir acciones españolas o europeas a tu cesta hace falta, además, que el bot tenga su historial de precios. Es el paso siguiente una vez que el ensayo confirme que tu cuenta las ofrece.

## Acciones sueltas: la cesta de favoritas

El modelo compra empresas a través de los fondos de índice, porque ninguna regla para elegir acciones ganó al índice cuando se probó (está en el panel y en `docs/RESEARCH.md`, paso 8). Si aun así quieres tener empresas concretas, hay una cesta aparte para eso. Está **apagada** hasta que la enciendas, y el panel enseña cada noche qué haría.

**Qué hace, encendida**

- Vigila tu lista (`config/favourites.json`): Nvidia, Tesla, Apple, Alphabet, Microsoft, Amazon, Meta y SpaceX.
- Tiene como máximo **4 empresas a la vez, un 2,5 % de la cuenta cada una**: un 10 % en total.
- Una empresa entra solo si su tendencia es alcista (probabilidad del 55 % o más, con el mismo filtro que usa el modelo) y sale cuando deja de serlo (por debajo del 45 %). Se decide una vez por semana, con el resto de la cartera.
- Una acción recién salida a bolsa espera 180 días: antes, el filtro no tiene con qué juzgarla. SpaceX cotiza desde el 12 de junio de 2026, así que podrá entrar a partir del 9 de diciembre.
- Solo usa dinero que el modelo deja en liquidez y nunca cambia lo que hace el modelo. Si el modelo necesita ese dinero, la cesta encoge.
- El cortacircuitos también la vende.

**Cómo se enciende:** en GitHub, *Settings → Secrets and variables → Actions → Variables → New repository variable*, nombre `HERMES_FAVOURITES`, valor `yes`. La ejecución siguiente fija el objetivo de la cesta. En una cuenta europea las compras aparecen, como las de los fondos, en el aviso de órdenes pendientes de colocar a mano (Convertir, de EUR a AAPLx, MSFTx...).

**Cómo se apaga:** borra la variable o ponla en `no`. Apagarla **no vende nada**: las acciones que tengas se quedan en la cuenta, valoradas pero fuera del modelo, hasta que las vendas tú. Lo mismo vale para una acción de la lista que compres por tu cuenta con la cesta apagada: el bot la ve, la cuenta en el valor de la cuenta y no la toca.

Con la cesta **encendida**, en cambio, las acciones de la lista las lleva la regla: si tienes una que la regla no quiere en ese momento (por ejemplo, una comprada por tu cuenta que está en tendencia bajista), el bot propondrá venderla.

**Cómo se cambia la lista:** edita `config/favourites.json` (el símbolo de la acción y su nombre). Solo sirven acciones que Kraken tenga tokenizadas y que de verdad se negocien; el panel dice de cada una si entra, si espera o por qué no. Los dos límites (`per_stock` y `max_stocks`) también están ahí; el programa rechaza el fichero si, multiplicados, pasan del 30 % de la cuenta, y si no puede leerlo sigue operando sin la cesta y lo avisa.

**Lo que conviene saber antes**

- No hay prueba de que esto gane al índice. Con estas siete empresas la regla habría ido muy bien desde 2016, pero la lista está escrita hoy, sabiendo cuáles ganaron: ese resultado no demuestra nada.
- Lo que sí se puede medir es cómo se comporta la regla. Cada empresa entra en la cesta menos de una vez al año. Las comisiones se llevan alrededor de un 0,8 % anual de la cesta por el libro de órdenes y un 2,5 % con Convertir, que es lo que paga una cuenta europea. La mayor caída de la cesta fue del 37 %, frente al 53 % de tener las siete siempre. Está en `research/12_favourites_basket.py`.
- Con una cuenta de 230 €, cada posición son unos 5 €. Los porcentajes son los mismos con cualquier tamaño.
- Anthropic no cotiza todavía. Cuando salga a bolsa y Kraken liste su acción, se añade a la lista y esperará sus 180 días como cualquier otra.
- **Bolsa española, europea, japonesa o china:** Kraken no tiene ninguna acción de esos mercados. De sus 177 acciones tokenizadas solo cuatro no son estadounidenses (ASML, Novo Nordisk, AstraZeneca y TSMC) y solo ASML y TSMC se negocian de verdad; sus fondos de Europa y del mundo pasan días sin una sola operación. Para esos mercados hace falta otro bróker con API.

## 5. Qué hacer con el bot antiguo

El Hermes Trader anterior vive en Docker en el portátil y lleva parado desde el 10 de noviembre. Antes de armar el nuevo:

1. **Apágalo del todo** (`docker compose down` en su carpeta) para que no vuelva a operar por su cuenta si enciendes el portátil.
2. **Entra en Kraken y mira las órdenes abiertas.** El bot dejó tres entradas a la espera (LINK, ETH, SOL). Si eran órdenes reales en Kraken, cancélalas.
3. Desactiva o borra su clave de API antigua.

Dos bots sobre la misma cuenta se pisarían.

## 6. Qué esperar

Sin adornos:

- **Rentabilidad razonable de esperar: en torno al 10 % anual**, con años negativos. En el examen a ciegas fueron +11,9 % anual con las comisiones del libro de órdenes y +10,9 % con lo que de verdad cuesta comprar los fondos a mano en una cuenta europea; el peor año, −5,5 %.
- **Con 233 € eso son unos 25 € al año.** El modelo no convierte poco dinero en mucho. Lo que hace es ganar de forma defendible y escalar: la misma lógica funciona igual con 2.000 o con 20.000 €.
- **Habrá rachas malas.** La peor caída histórica fue del −17 %. Es normal estar meses por debajo del máximo.
- **No bate a todo.** Entre 2022 y 2026 una simple mezcla de bolsa y oro ganó más. La ventaja de Hermes fue caer menos.
- **Nadie puede garantizar beneficios.** Quien lo prometa, miente.

## 7. Perfiles

| | Equilibrado | Crecimiento |
|---|---|---|
| Objetivo de volatilidad | 15 % | 20 % |
| Reparto de riesgo (acciones / activos reales / cripto) | ⅓ · ⅓ · ⅓ | 30 % · 20 % · 50 % |
| Rentabilidad anual, examen a ciegas | 11,9 % | 12,1 % |
| Peor caída, examen a ciegas | −11,3 % | −13,2 % |

En el examen a ciegas, el perfil de crecimiento asumió más riesgo sin ganar más. El equilibrado es el recomendado.

## 8. TradingView

`tradingview/hermes_trend.pine` dibuja en cualquier gráfico la probabilidad de tendencia que usa el modelo. En TradingView: **Pine Editor → pegar el contenido → Add to chart**. Sirve para ver la señal sobre cualquier activo, no solo los de la cartera. Es una aproximación fiel pero simplificada: las decisiones las toma el código de este repositorio.

## 9. Si algo falla

- **El panel no se actualiza:** mira la pestaña Actions. GitHub te manda un correo cuando una ejecución falla.
- **"data problems" en el registro:** Kraken no respondió para algún mercado. El modelo congela ese mercado por un día y sigue; si faltan muchos, no opera.
- **GitHub desactiva los flujos programados tras 60 días sin actividad** en el repositorio. Como el modelo guarda su estado a diario, eso no ocurre mientras funcione.
