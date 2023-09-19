from recbole.trainer import Trainer

class NewTrainer(Trainer):

    def __init__(self, config, model):
        super(NewTrainer, self).__init__(config, model)

    def _train_epoch(self, train_data, epoch_idx, show_progress=True):
        self.model.train()
        total_loss = 0.

        if epoch_idx % 2 == 0:
            for batch_idx, interaction in enumerate(train_data):
                interaction = interaction.to(self.device)
                self.optimizer.zero_grad()
                loss = self.model.calculate_loss(interaction)
                self._check_nan(loss)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
        else:
            for batch_idx, interaction in enumerate(train_data):
                interaction = interaction.to(self.device)
                self.optimizer.zero_grad()
                loss = self.model.calculate_loss(interaction)
                self._check_nan(loss)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
        return total_loss
