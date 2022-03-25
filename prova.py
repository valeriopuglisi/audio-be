from zipfile import ZipFile

from datetime import datetime
import pathlib
import os
# datetime object containing current date and time
now = datetime.now()

print("now =", now)

# dd/mm/YY H:M:S
dt_string = now.strftime("%d_%m_%Y__%H_%M_%S")
report_id = "report__" + dt_string
print("report_id: ", report_id)
report_name = report_id + '.zip'
REPORTS_PATH = os.path.join(os.getcwd(), "reports")
pathlib.Path(REPORTS_PATH).mkdir(parents=True, exist_ok=True)
report_path = os.path.join(REPORTS_PATH, report_name)

# Create a ZipFile Object
with ZipFile(report_path, 'w') as zipObj2:
   # Add multiple files to the zip
   zipObj2.write("pipelines\VAD + Separation .yml")