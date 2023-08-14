# Algoritmické řešení s pomocí hranové detekce

Popis skriptů:
- [algorithmic.py](./algorithmic.py) je kompletní algoritmické řešení včetně preprocessingu i postprocessingu
- [intrinsic_calibration.py](./intrinsic_calibration.py) vytvoří na základě kalibračního datasetu v [resources](../resources) kalibrační soubor s vnitřními parametry kamery a koeficienty pro korekci
- [hsv_thresholder.py](./hsv_thresholder.py) je skript pro výběr konkrétní barvy pomocí prahování v HSV barevném modelu H(hue) odstín, S(saturation) saturace, V(value) jas