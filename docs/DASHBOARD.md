# Dashboard (Streamlit)

## Requisitos
- PostgreSQL accesible vía `DATABASE_URL`.
- Dependencias instaladas: `pip install -r requirements.txt`.

## Ejecución
```bash
export DATABASE_URL=postgresql://user:password@localhost:5432/mydb
streamlit run dashboard/app.py
```

## Páginas
- **Overview**: KPIs (ventas, unidades, #productos, stock, margen), serie mensual, top productos/categorías/materiales.
- **Materiales/Categorías**: rankings y meses con más ventas (global y por material/categoría).
- **Productos**: histórico por producto (unidades/revenue) y margen estimado si hay costo/precio.
- **Predicción**: filtros por producto/horizonte/modelo/run_tag, gráfico histórico vs yhat, tabla de métricas.
- **Video analytics (opcional)**: conteos por clase y por tiempo desde `tracks`.

Cada tabla/gráfico tiene botón para exportar CSV.
