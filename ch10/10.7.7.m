% ref: https://blog.csdn.net/qiudw/article/details/8615830
fd = fopen('iris.data');
data = textscan(fd, '%f%f%f%f%s', 'delimiter', ',');
x = cell2mat([data(1), data(2), data(3), data(4)]);
class = data(5);
fclose(fd);
y = zeros(150, 1);
y(strcmp(class{1}, 'Iris-setosa')) = 1;
y(strcmp(class{1}, 'Iris-versicolor')) = 2;
y(strcmp(class{1}, 'Iris-virginica')) = 3;

ensemble = fitensemble(x, y, 'AdaBoostM2', 100, 'Tree', 'KFold', 10);
ensemble.kfoldLoss

ctree1 = fitctree(x, y, 'MaxNumSplits', 1, 'KFold', 10);
ctree1.kfoldLoss

ctree2 = fitctree(x, y, 'KFold', 10, 'MaxNumSplits', 10);
ctree2.kfoldLoss
