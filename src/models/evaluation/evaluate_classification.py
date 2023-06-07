

# -----------------------------  EVALUATING ----------------------------------
def evaluate_model(trainer):
    print("Evaluating model: " + trainer.save_model_path + trainer.model_name + ".h5")

    trainer.model.load_weights(trainer.save_model_path + trainer.model_name + ".h5")
    batch_generator = trainer.generator.validation
    pred_test = trainer.model.evaluate(batch_generator,
                               steps=batch_generator.steps_per_epoch,
                               return_dict=True
                               )

    print(f'{"Metric":<12}{"Value"}')
    for metric, value in pred_test.items():
        print(f'{metric:<12}{value:<.4f}')
