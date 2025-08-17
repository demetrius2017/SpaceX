# SpaceX Risk/Growth Dashboard

Готовая утилита для мониторинга SpaceX: собирает новости/контракты/регуляторные события, отмечает красные флаги и считает индекс, выдаёт рекомендацию (увеличивать/держать/подрезать/снижать).

## Быстрый старт
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Сбор и построение панели
python monitor.py run --days 45

# Быстрый отчёт в консоль (без HTML)
python monitor.py report --days 45
```
Открой `monitor_output/dashboard.html` в браузере.

## Конфигурация

## Ежедневный запуск (macOS, launchd)

В репозитории добавлены:
- `scripts/run_monitor_daily.sh` — обёртка, активирует venv, читает `.env` и запускает `python monitor.py run --days 45 --telegram` с логированием в `monitor_output/logs/`.
- `scripts/com.github.spacex.monitor.daily.plist` — шаблон launchd-агента для ежедневного запуска в 09:00.

Как подключить:
1) Отредактируйте `scripts/com.github.spacex.monitor.daily.plist` (WorkingDirectory, время, переменные при необходимости).
2) Скопируйте в `~/Library/LaunchAgents/` и загрузите:
	- `cp scripts/com.github.spacex.monitor.daily.plist ~/Library/LaunchAgents/`
	- `launchctl unload ~/Library/LaunchAgents/com.github.spacex.monitor.daily.plist 2>/dev/null || true`
	- `launchctl load ~/Library/LaunchAgents/com.github.spacex.monitor.daily.plist`
3) Проверить статус: `launchctl list | grep com.github.spacex.monitor.daily`
4) Принудительный запуск для теста: `launchctl kickstart -k gui/$(id -u)/com.github.spacex.monitor.daily`

Переменные окружения можно задать через `.env` в корне проекта (OPENAI_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID), либо в самом plist.

## Что значит «агрегация для конкурентов»?

В коде включена защита от всплесков дублей новостей по конкурентам (в частности, Amazon Kuiper):
- После применения правил мы группируем события категории `Competitors/Kuiper*` по ключу «тэг + дата (ДЕНЬ)» и оставляем максимум одно событие в сутки на тэг.
- Это предотвращает искусственное увеличение «риска» из‑за множества почти идентичных заметок в один день.
- Механизм легко расширить на другие конкурирующие источники, если понадобится.

## Источники (по умолчанию)
- NASA News Releases (RSS) — подтверждение/задержки по HLS/Artemis
- NASA Artemis (RSS)
- DoD Contract Announcements (RSS) + HTML fallback
- Space Systems Command (RSS)
- FAA (RSS + HTML страница Starship)
- FCC Daily Digest/Headlines (RSS)
- SpaceX Updates/Launches (HTML)
- NASASpaceflight (RSS)
- Google News (тендерная оценка; Project Kuiper)
- T‑Mobile Newsroom (RSS)
- CA Coastal Commission (RSS)

## Индекс и правила
- Положительные веса: лицензии FAA/FONSI, победы/заказы по NSSL/Starshield, прогресс HLS, запуск D2C (T‑Mobile).
- Отрицательные: аварии/grounding Starship, проигрыш/отмена NSSL, задержки HLS, регуляторные блоки, негативная тендерная динамика, усиление Kuiper.

Пороговые значения по умолчанию: см. `config.yml`.

## USAspending API
Скрипт берёт новые награждения (DoD/NASA) для **SPACE EXPLORATION TECHNOLOGIES CORP.** за последние 120 дней и добавляет их как события.

## Telegram Bot и рассылка (без вебхука)

- `bot_poll.py` — Telegram-бот с long polling (Heroku worker). Реагирует на `/start`: регистрирует подписчика, отправляет последнее сообщение или свежую сводку.
- `db.py` — хранит подписчиков и историю индексов/сообщений (SQLite локально, Postgres на Heroku через `DATABASE_URL`).
- `scripts/daily_broadcast.sh` — ежедневная рассылка всем подписчикам, добавляет Δ индекса по сравнению с предыдущим днём.

Heroku:
1) В Procfile настроен `worker: python bot_poll.py`.
2) Установите Config Vars: `TELEGRAM_BOT_TOKEN`, (опц.) `OPENAI_API_KEY`, `DATABASE_URL` (Postgres), `BROADCAST_DAYS` (опц.).
3) Включите worker-процесс: `heroku ps:scale worker=1`.
4) Добавьте Heroku Scheduler: команда `bash scripts/daily_broadcast.sh` раз в сутки.

## Деплой на Heroku (Web + Mini App)

В репозитории есть Procfile с тремя процессами: `web` (дашборд), `worker` (бот) и `release`.

1) Требования
	- Аккаунт Heroku, Heroku CLI, подключённый GitHub.
	- Buildpacks: Python (по умолчанию); runtime.txt задаёт версию Python.

2) Переменные окружения (Heroku Config Vars)
	- OPENAI_API_KEY — ключ с правом `model.request`. При использовании project‑key добавить `OPENAI_PROJECT`.
	- TELEGRAM_BOT_TOKEN — токен бота.
	- TELEGRAM_CHAT_ID — (опционально) чат для отправки отчётов.
	- PUBLIC_URL — публичный HTTPS‑URL, ведущий на каталог `monitor_output/` (например, `https://your-app.herokuapp.com/`).
	- TELEGRAM_USE_WEBAPP — `1` чтобы кнопка открывала Mini App внутри Telegram (требует HTTPS и, возможно, настройку домена у BotFather).
	- DATABASE_URL — (опционально) Postgres для бота/истории.

3) Развёртывание
	- Подключите репозиторий к приложению Heroku или используйте `git push heroku main`.
	- Включите процессы:
	  - `heroku ps:scale web=1 worker=1`
	- Проверьте логи: `heroku logs -t`.

4) Как это работает
	- Процесс `web` запускает: `python monitor.py run --serve --host 0.0.0.0 --port $PORT --public-url $PUBLIC_URL`.
	  - Генерирует `monitor_output/dashboard.html` и поднимает HTTP‑сервер, отдающий `/`, `/dashboard.html`, `/health`.
	  - Если заданы TELEGRAM_* и `--telegram`, отправит сообщение с кнопкой “Открыть дашборд”.
	- Процесс `worker` запускает `bot_poll.py` (long polling).

5) GitHub Environments / Secrets
	- Можно хранить секреты в GitHub Secrets и деплоить через GitHub Actions:
	  - Settings → Secrets and variables → Actions → New repository secret: `OPENAI_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `HEROKU_API_KEY`.
	  - Создайте Workflow, который при пуше в main деплоит на Heroku и проставляет Config Vars командой `heroku config:set ...`.
	- Также можно использовать GitHub Environments (prod/staging) с разными наборами секретов.

6) Мини‑приложение Telegram (Mini App)
	- Убедитесь, что `PUBLIC_URL` — HTTPS и отдаёт `dashboard.html` по `/` (в коде настроен редирект `/` → `/dashboard.html`).
	- Включите `TELEGRAM_USE_WEBAPP=1`, чтобы отправлялась web_app‑кнопка.
	- По необходимости установите домен у BotFather (/setdomain) для Web Apps.


## Настройка под себя
- Добавляй/удаляй источники в `config.yml`.
- Меняй веса правил под свою модель риска/дохода.
- Отмечай ключевые флаги в `checklist.yaml` (true/false).

---

Автор: GPT‑5 Pro (ведущий разработчик). Гарантирую читабельность и поддержку кода в пределах данного репозитория.
# SpaceX
