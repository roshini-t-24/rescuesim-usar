import json
a = json.load(open("results/serverless_eval.json"))["summary"]
b = json.load(open("results/serverless_eval_new.json"))["summary"]
print(f'{"policy":<9} {"rescue% old/new":>17} {"loss% old/new":>15} {"Wh/resc old/new":>17}')
for p in ("greedy", "static", "adaptive"):
    print(f'{p:<9} {a[p]["rescue_rate_pct"]["mean"]:>7.1f}/{b[p]["rescue_rate_pct"]["mean"]:<8.1f} '
          f'{a[p]["robot_loss_rate_pct"]["mean"]:>6.1f}/{b[p]["robot_loss_rate_pct"]["mean"]:<7.1f} '
          f'{a[p]["energy_efficiency_wh_per_rescued"]:>8.1f}/{b[p]["energy_efficiency_wh_per_rescued"]:<8.1f}')
