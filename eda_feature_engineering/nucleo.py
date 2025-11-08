import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eda_feature_engineering import abonados, accesos, economia, servicios, archivo_final

def main():
    print("Ejecutando análisis de abonados...")
    df_abonados = abonados.run()
    print(df_abonados)
      
    print("\nEjecutando análisis de accessos...")
    df_accessos = accesos.run()
    print(df_accessos)

    
    print("\nEjecutando análisis de economia...")
    df_economia = economia.run()
    print(df_economia)
   
    print("\nEjecutando análisis de servicios..")
    df_servicios = servicios.run()
    print(df_servicios)
   

    # Por ejemplo, solo un mensaje:
    print("\n✅ Análisis completos y archivos generados.")

    print("\nProcesando el dataframe final para el modelo....")
    df_pre_modelo = archivo_final.run()
    print(df_pre_modelo)


    print("✅ Archivo final generado. Ya se puede pasar a crear el modelo!")
if __name__ == "__main__":
    main()