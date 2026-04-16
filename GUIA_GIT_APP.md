# Cómo llevar estos cambios a tu Git

Esta guía te deja **dos formas** para tener la app actualizada en tu repositorio.

## Opción A (recomendada): subir tus cambios directo

Si ya tienes los archivos modificados en tu carpeta local:

```bash
git add app/main.py app/app.py app/shared.py app/pages/01_prediction.py app/pages/02_batch_prediction.py app/pages/03_eda.py app/pages/04_shap_values.py app/pages/05_model_comparison.py app/requirements.txt requirements.txt

git commit -m "Refactor app multipágina: test individual, CSV, EDA, SHAP y comparación train/test"

git push origin <tu-rama>
```

Luego abre un Pull Request desde `<tu-rama>` hacia `main` (o la rama que uses como base).

---

## Opción B: traer exactamente el commit ya hecho

Si quieres traer tal cual el cambio que ya quedó hecho, usa `cherry-pick` del commit:

```bash
git fetch origin

git checkout <tu-rama>

git cherry-pick 29b2e32

git push origin <tu-rama>
```

> Si te sale conflicto en `cherry-pick`:
> 1) resuelve archivos,
> 2) `git add .`,
> 3) `git cherry-pick --continue`.

---

## Validación rápida antes de push

```bash
python -m py_compile app/main.py app/app.py app/shared.py app/pages/01_prediction.py app/pages/02_batch_prediction.py app/pages/03_eda.py app/pages/04_shap_values.py app/pages/05_model_comparison.py
```

---

## Ejecutar la app

Desde la raíz del proyecto:

```bash
streamlit run app/main.py
```

En el menú lateral de Streamlit verás las páginas:
1. Test del modelo (individual)
2. Test del modelo por CSV
3. EDA
4. SHAP values
5. Comparación de modelos

---

## Nota sobre SHAP y modelo lineal

Sí: **SHAP funciona con modelos lineales** y es totalmente válido para explicar contribuciones por variable.
