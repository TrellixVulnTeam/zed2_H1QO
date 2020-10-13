
import csv
def export_list_csv(export_list, csv_dir):

    with open(csv_dir, "w") as f:
        writer = csv.writer(f, lineterminator='\n',delimiter=' ',)

        if isinstance(export_list[0], list): #多次元の場合
            writer.writerows(export_list)

        else:
            writer.writerow(export_list)
export_list_csv([1,2,2],'aa.txt')
print(round(0.12344989797199, 10))
