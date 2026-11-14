# Guía de puesta en marcha

Todo lo necesario para dejar Hermes Quant funcionando solo, cada noche, sin depender de ningún ordenador.

## 1. Lo que hay y lo que hace

Cada noche, poco después de las 00:00 UTC, GitHub ejecuta el modelo:

1. Descarga las velas del día de Kraken (12 criptomonedas, 8 fondos tokenizados y el euro/dólar).
2. Recalcula volatilidades, tendencias, régimen de mercado y correlaciones.
3. Una vez por semana fija los pesos objetivo; el resto de días solo opera si algún peso se ha desviado lo bastante como para compensar la comisión.
4. Ejecuta las órdenes (simuladas por defecto), guarda el estado en el propio repositorio y actualiza el panel.

No hay servidor, ni Docker, ni coste de IA. Si el portátil está apagado, da igual.

## 2. Publicarlo (una sola vez, 5 minutos)

1. Sube la carpeta a un repositorio de GitHub llamado `hermes-quant`.
2. En el repositorio: **Settings → Actions → General → Workflow permissions → Read and write permissions**. Guarda.
3. **Settings → Pages → Build and deployment → Source: GitHub Actions.**
4. **Actions → Hermes daily run → Run workflow** para lanzar la primera ejecución a mano.

A los dos minutos el panel estará en `https://TU_USUARIO.github.io/hermes-quant/`.

GitHub Pages es gratis en repositorios públicos. Si el repositorio es privado, el modelo funciona igual, pero el panel no se publica hasta que lo hagas público (o hasta que conectes la carpeta `site/` a Netlify).

## 3. Avisos por Telegram (opcional)

1. En Telegram, habla con `@BotFather`, escribe `/newbot` y guarda el token.
2. Escribe cualquier cosa a tu bot nuevo y abre `https://api.telegram.org/botTOKEN/getUpdates` para ver tu `chat.id`.
3. En GitHub: **Settings → Secrets and variables → Actions → New repository secret**. Crea `TELEGRAM_BOT_TOKEN` y `TELEGRAM_CHAT_ID`.

Recibirás un mensaje diario con el valor de la cartera, las órdenes y las posiciones.

## 4. Pasar a dinero real

Hazlo solo después de ver el modo simulado funcionar unas semanas. Son tres pasos, y cada uno es un candado distinto.

**Candado 1: las claves.** En Kraken: **Settings → API → Create key**, con estos permisos y ninguno más:

- Query funds
- Query open orders & trades
- Create & modify orders
- Cancel & close orders

**Nunca marques "Withdraw funds".** Con esa casilla sin marcar, nadie que robe la clave puede sacar dinero.

En GitHub crea dos secretos: `KRAKEN_API_KEY` y `KRAKEN_API_SECRET`.

**Candado 2: el modo.** En **Settings → Secrets and variables → Actions → Variables** crea `HERMES_MODE` con valor `live`. A partir de aquí el modelo lee tu saldo real y **envía cada orden a Kraken solo para validarla**: Kraken comprueba que es correcta y la descarta. No se mueve ni un céntimo. Revisa el registro de la ejecución (pestaña Actions) durante varios días: debe mostrar órdenes con estado `validated` y sin errores.

**Candado 3: armarlo.** Cuando el registro te convenza, crea la variable `HERMES_LIVE_ARMED` con valor `yes`. Ahora sí opera.

Dos límites adicionales que puedes ajustar:

- `HERMES_MAX_ORDER_EUR` (por defecto 100): ninguna orden, conversiones de divisa incluidas, supera esa cantidad.
- `HERMES_PROFILE`: `balanced` (por defecto) o `growth`.

> El adaptador de Kraken se ha probado contra un exchange simulado y contra el ejemplo oficial de firma de Kraken, pero no contra una cuenta con fondos. Por eso existe el candado 2: úsalo.

### Cómo se envían las órdenes

Todas son órdenes límite con un tope del 0,5 % sobre el precio de referencia y de tipo "inmediata o cancelada": o se ejecutan al momento a un precio aceptable, o no se ejecutan. No queda ninguna orden viva en el libro. Después el modelo lee de Kraken lo que de verdad se ha ejecutado; nunca lo da por hecho.

### Si ingresas o retiras dinero

El modelo mide las caídas desde el máximo de la cuenta. Un retiro parecería una pérdida. Tras cualquier ingreso o retiro, crea la variable `HERMES_REBASE` con valor `yes`, deja pasar una ejecución y bórrala.

### El cortacircuitos

Si la cuenta cae un 25 % desde su máximo, el modelo vende todo, se queda en liquidez y avisa. No vuelve a operar hasta que alguien edite `data/live/state.json` y ponga `"halted": false`. Está pensado para que un fallo, del modelo o del mercado, no se convierta en una ruina.

## 5. Qué hacer con el bot antiguo

El Hermes Trader anterior vive en Docker en el portátil y lleva parado desde el 10 de noviembre. Antes de armar el nuevo:

1. **Apágalo del todo** (`docker compose down` en su carpeta) para que no vuelva a operar por su cuenta si enciendes el portátil.
2. **Entra en Kraken y mira las órdenes abiertas.** El bot dejó tres entradas a la espera (LINK, ETH, SOL). Si eran órdenes reales en Kraken, cancélalas.
3. Desactiva o borra su clave de API antigua.

Dos bots sobre la misma cuenta se pisarían.

## 6. Qué esperar

Sin adornos:

- **Rentabilidad razonable de esperar: en torno al 10 % anual**, con años negativos. En el examen a ciegas fueron +11,9 % anual; el peor año, −5,5 %.
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
