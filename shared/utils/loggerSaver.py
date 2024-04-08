import os
import torch


class LoggerSaver:

    def __init__(self):
        self.current_epoch = -1
        self.lines = []
        self.output_file = None
        self.best_metrics = {'acc': 0, 'P': 0, 'R': 0, 'F1': 0}
        self.best_model_name = {'acc': 'none', 'P': 'none', 'R': 'none', 'F1': 'none'}

    def add_result(self, accuracy_t, loss_t, accuracy, p, r, f1, loss, train_time, valid_time):
        line = "%d,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.2f,%.2f\n" % (
            self.current_epoch, accuracy_t, loss_t, accuracy, p, r, f1, loss, train_time, valid_time
        )
        self.lines.append(line)

        if self.output_file is not None:
            with open(self.output_file, "a+") as outf:
                outf.write(line)

    def set_save_file(self, filename):
        self.output_file = filename
        if os.path.exists(filename):
            return
        if self.output_file is not None:
            with open(self.output_file, "w+") as outf:
                outf.write("Epoch,Accuracy(Train),Loss(Train),Accuracy,Precision,Recall,F1,Loss,Train time(s),"
                           "Valid Time\n")

    def reset(self):
        self.lines = []
        self.output_file = None

    def get_full_outpath(self):
        return os.getcwd() + "/" + self.output_file

    def saveModel(self, filename, model, epoch):
        params = {
            'model': model.state_dict(),
            'epoch': epoch,
            'best_model_name': self.best_model_name
        }
        torch.save(params, filename)

    def save(self, save_dir, model, metrics):

        for m in self.best_metrics:
            if (metrics[m] >= self.best_metrics[m]):
                if os.path.exists(os.path.join(save_dir, self.best_model_name[m])):
                    os.remove(os.path.join(save_dir, self.best_model_name[m]))

                self.best_model_name[m] = "Best(%s)-ep%03d.pth" % \
                                          (m, self.current_epoch)
                filename = os.path.join(save_dir, self.best_model_name[m])
                self.saveModel(filename, model, self.current_epoch + 1)
                self.best_metrics[m] = metrics[m]

        if os.path.exists(os.path.join(save_dir, "model.pth")):
            os.remove(os.path.join(save_dir, "model.pth"))
        filename = os.path.join(save_dir, "model.pth")
        self.saveModel(filename, model, self.current_epoch + 1)


logger_saver = LoggerSaver()
