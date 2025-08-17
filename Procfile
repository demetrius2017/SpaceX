web: python monitor.py run --serve --host 0.0.0.0 --port $PORT --gpt-timeout 60 --gpt-max-items 40 --public-url $PUBLIC_URL
worker: python bot_poll.py
release: python -c "print('release phase ok')"
