nohup xvfb-run -s "-screen 0 1400x900x24" python3 -m trainer.train params_sonic.json | ts >> 'log1.out' 2>&1
