from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Optional
from core.AG_engine import AGEngine, AGType, run_algorithm
from schemas.data_models import DatosAG, DatosAGConfig

router = APIRouter()


@router.websocket("/ws/ag")
async def websocket_ag_endpoint(websocket: WebSocket):
    await websocket.accept()

    engine = AGEngine()

    async def send_progress(message):
        try:
            await websocket.send_json(message)
        except Exception as e:
            # Si falla el envío, lanzamos un error que rompa la ejecución del algoritmo
            raise RuntimeError("Cliente desconectado. Deteniendo algoritmo.")

    engine.set_websocket(send_progress)

    try:
        # EL CAMBIO CLAVE: Este bucle mantiene viva la conexión
        while True:
            # El código se pausará aquí esperando instrucciones del frontend
            data = await websocket.receive_json()
            
            algorithm_type = AGType(data.get("algorithm", "feature_selection"))
            config = data.get("config", {})

            engine.load_data(config.get("dataset", "breast_cancer"))

            # Ejecutamos el algoritmo solicitado
            if algorithm_type == AGType.FEATURE_SELECTION:
                result = await engine.run_feature_selection(
                    population_size=config.get("population_size", 20),
                    generations=config.get("generations", 15),
                    mutation_rate=config.get("mutation_rate", 0.05),
                    tournament_size=config.get("tournament_size", 3),
                )
            elif algorithm_type == AGType.HYPERPARAMETER_OPT:
                result = await engine.run_hyperparameter_optimization(
                    population_size=config.get("population_size", 10),
                    generations=config.get("generations", 10),
                    mutation_rate=config.get("mutation_rate", 0.2),
                )
            elif algorithm_type == AGType.NEUROEVOLUTION:
                result = await engine.run_neuroevolution(
                    population_size=config.get("population_size", 15),
                    generations=config.get("generations", 12),
                    mutation_rate=config.get("mutation_rate", 0.3),
                    max_layers=config.get("max_layers", 5),
                    max_neurons=config.get("max_neurons", 128),
                )
            
            # Opcional: Avisar al frontend que este trabajo en particular ya terminó
            await websocket.send_json({
                "type": "finished", 
                "message": "Algoritmo completado", 
                "result": result
            })

    except WebSocketDisconnect:
        # Aquí entra si el usuario cierra la pestaña de forma normal
        print("El cliente se ha desconectado.")
    except Exception as e:
        error_msg = str(e)
        # Si el error fue provocado a propósito por la desconexión, solo lo imprimimos
        if error_msg == "Cliente desconectado. Deteniendo algoritmo.":
            print(error_msg)
        else:
            # Si fue un error real del algoritmo, se lo enviamos al frontend (si sigue ahí)
            try:
                await websocket.send_json({"type": "error", "message": error_msg})
            except:
                pass


@router.post("/feature-selection")
async def run_feature_selection(datos: Optional[DatosAGConfig] = None):
    config = datos.dict() if datos else {}

    async def progress_callback(message):
        pass

    result = await run_algorithm(
        AGType.FEATURE_SELECTION,
        websocket_callback=progress_callback,
        population_size=config.get("population_size", 20),
        generations=config.get("generations", 15),
        mutation_rate=config.get("mutation_rate", 0.05),
        tournament_size=config.get("tournament_size", 3),
    )
    return result


@router.post("/hyperparameter-optimization")
async def run_hyperparameter_optimization(datos: Optional[DatosAGConfig] = None):
    config = datos.dict() if datos else {}

    async def progress_callback(message):
        pass

    result = await run_algorithm(
        AGType.HYPERPARAMETER_OPT,
        websocket_callback=progress_callback,
        population_size=config.get("population_size", 10),
        generations=config.get("generations", 10),
        mutation_rate=config.get("mutation_rate", 0.2),
    )
    return result


@router.post("/neuroevolution")
async def run_neuroevolution(datos: Optional[DatosAGConfig] = None):
    config = datos.dict() if datos else {}

    async def progress_callback(message):
        pass

    result = await run_algorithm(
        AGType.NEUROEVOLUTION,
        websocket_callback=progress_callback,
        population_size=config.get("population_size", 15),
        generations=config.get("generations", 12),
        mutation_rate=config.get("mutation_rate", 0.3),
        max_layers=config.get("max_layers", 5),
        max_neurons=config.get("max_neurons", 128),
    )
    return result
