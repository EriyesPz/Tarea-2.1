from src.procesos import cargar_datos, procesar_datos
from src.modelos  import entrenar_modelo


def main() -> None:
    df = cargar_datos("data/housing.csv")
    df = procesar_datos(df)

    modelo, r2_train, r2_test, rmse = entrenar_modelo(df)

    print("\nRESULTADOS:")
    print(f"Score de entrenamiento: {r2_train:.4f}")
    print(f"Score de prueba:       {r2_test:.4f}")
    print(f"RMSE:                  {rmse:,.2f}")


if __name__ == "__main__":
    main()
