nohup xvfb-run -s "-screen 0 1400x900x24" python3 -u -m trainer.train params_sonic.json >> 'log.out' 2>&1 &
