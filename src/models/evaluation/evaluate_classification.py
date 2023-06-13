from src.visualization.visual_model import visual_model
from src.visualization.confusion_matrix import confusion_matrix_and_report

# -----------------------------  EVALUATING ----------------------------------
def evaluate_model(trainer, reports_path, conf_matrix, model_layout):
    print("Evaluating model: " + trainer.model_path + trainer.model_name)

    trainer.model.load_weights(trainer.model_path + trainer.model_name)
    batch_generator = trainer.generator.validation
    '''
    score = trainer.model.evaluate(batch_generator,
                               steps=batch_generator.steps_per_epoch,
                               return_dict=True
                               )
    print(f'{"Metric":<12}{"Value"}')
    for metric, value in score.items():
        print(f'{metric:<12}{value:<.4f}')
    
    '''
    # plot confusion matrix and save classification report to src/reports
    if conf_matrix:
        confusion_matrix_and_report(trainer, batch_generator, reports_path)

    # plot model layout and save to src/reports/figures
    if model_layout:
        visual_model(trainer.model, reports_path, trainer.model_arch)
