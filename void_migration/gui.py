from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from kivymd.app import MDApp as App
from kivymd.uix.boxlayout import BoxLayout
from kivymd.uix.selectioncontrol import MDCheckbox as CheckBox
from kivymd.uix.label import MDLabel as Label
from kivymd.uix.slider import MDSlider as Slider
from kivymd.uix.textfield import MDTextField as TextInput
from kivymd.uix.menu import MDDropdownMenu as DropdownMenu
from kivymd.uix.dropdownitem import MDDropDownItem as DropDownItem
from kivymd.uix.button.button import MDRaisedButton as Button
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserListView

# from kivy.cache import Cache
from kivy.config import Config
from kivy.logger import Logger, LOG_LEVELS
import void_migration.params as params
from void_migration.main import time_march, init


import sys
import os
import signal
import multiprocessing
from functools import partial
import csv

os.environ["KIVY_NO_ARGS"] = "1"
os.environ["KIVY_NO_CONSOLELOG"] = "1"

# Set log level to warning
Config.set("kivy", "log_level", "warning")
Logger.setLevel(LOG_LEVELS["warning"])


def run_time_march(p, *args):
    p.set_defaults()
    time_march(p)


def run_init(p, *args):
    p.set_defaults()
    init(p)


class VoidMigrationApp(App):
    def __init__(self, data, p, **kwargs):
        super().__init__(**kwargs)
        self.data = data
        self.p = p
        self.halt = False
        self.queue = multiprocessing.Queue()
        self.queue2 = multiprocessing.Queue()
        self.process = None
        self.stop_event = multiprocessing.Event()
        self.menus = {}
        self.p.queue = self.queue
        self.p.queue2 = self.queue2
        self.p.stop_event = self.stop_event

    def build(self):
        self.title = "Void Migration"
        main_layout = BoxLayout(orientation="horizontal")
        param_layout = BoxLayout(orientation="vertical", size_hint_x=0.3, padding=20)

        for key, limits in self.data["gui"].items():
            value = getattr(self.p, key)

            param_layout.add_widget(Label(text=limits["title"]))
            if limits["dtype"] == "bool":
                input_widget = CheckBox(active=value)
                input_widget.bind(active=partial(self.update_param, key=key))
            elif limits["dtype"] == "int":
                input_widget = Slider(min=limits["min"], max=limits["max"], step=limits["step"], value=value)
                input_widget.bind(value=partial(self.update_param, key=key))
            elif limits["dtype"] == "float":
                input_widget = Slider(min=limits["min"], max=limits["max"], step=limits["step"], value=value)
                input_widget.bind(value=partial(self.update_param, key=key))
            elif limits["dtype"] == "str":
                input_widget = TextInput(text=value)
                input_widget.bind(text=partial(self.update_param, key=key))
            elif limits["dtype"] == "select":
                input_widget = DropDownItem()
                menu_items = []
                for i, option in enumerate(limits["options"]):
                    if "labels" in self.data["gui"][key]:
                        label = self.data["gui"][key]["labels"][i]
                    else:
                        label = option
                    menu_items += [
                        {
                            "text": label,
                            "viewclass": "OneLineListItem",
                            "on_release": lambda option=option, label=label, key=key: self.menu_callback(
                                option, label, key
                            ),
                        }
                    ]
                dropdown_menu = DropdownMenu(
                    caller=input_widget,
                    items=menu_items,
                    width_mult=4,
                )
                self.menus[key] = dropdown_menu
                input_widget.bind(on_release=lambda x, k=key: self.menus[k].open())

                i = limits["options"].index(value)
                if "labels" in self.data["gui"][key]:
                    label = self.data["gui"][key]["labels"][i]
                else:
                    label = option
                input_widget.set_item(label)
            else:
                raise ValueError(f"Unsupported type: {type(value)} for key: {key}")
            param_layout.add_widget(input_widget)
            setattr(self, f"input_{key}", input_widget)

        buttons = BoxLayout(
            orientation="horizontal",
            size_hint_x=1.0,
            size_hint_y=None,
            height=80,
            spacing=20,
            # padding=[50, 0],
        )

        run_button = Button(text="Start", size_hint_x=0.5, size_hint_y=None, height=80)
        run_button.bind(on_press=self.start_time_march)
        buttons.add_widget(run_button)

        stop_button = Button(text="Stop", size_hint_x=0.5)
        stop_button.bind(on_press=self.stop_time_march)
        buttons.add_widget(stop_button)

        save_state_button = Button(text="Save state", size_hint_x=0.5)
        save_state_button.bind(on_press=lambda x: self.save_state())
        # buttons.add_widget(save_state_button)

        load_state_button = Button(text="Load state", size_hint_x=0.5)
        load_state_button.bind(on_press=lambda x: self.load_state())
        # buttons.add_widget(load_state_button)

        charge_button = Button(text="Enter charge/discharge", size_hint_x=0.5)
        charge_button.bind(on_press=lambda x: self.load_charge_discharge())
        # buttons.add_widget(charge_button)

        img_layout = BoxLayout(orientation="vertical")
        img_layout.add_widget(buttons)

        self.img = Image(allow_stretch=True, keep_ratio=True)
        img_layout.add_widget(self.img)

        # Add parameter and image layouts to the main layout
        main_layout.add_widget(param_layout)
        main_layout.add_widget(img_layout)

        Clock.schedule_interval(lambda dt: self.update_image(), 0.01)  # Start watching image directory

        run_init(self.p)

        return main_layout

    def menu_callback(self, option, label, key):
        dropdown_item = getattr(self, f"input_{key}", None)
        if isinstance(dropdown_item, DropDownItem):
            dropdown_item.set_item(label)
            self.update_param(option, key=key)
            self.menus[key].dismiss()

    def update_param(self, instance, *args, **kwargs):
        key = kwargs.get("key")
        if isinstance(instance, str):
            value = instance
        if isinstance(instance, CheckBox):
            value = instance.active
        elif isinstance(instance, DropDownItem):
            value = instance.current_item
        elif isinstance(instance, TextInput):
            value = instance.text
            if value.isdigit():
                value = int(value)
            else:
                try:
                    value = float(value)
                except ValueError:
                    pass
        elif isinstance(instance, Slider):
            value = instance.value

        setattr(self.p, key, value)
        print(f"Updated {key} to {value}")

        if key in ["view"]:
            self.queue2.put({key: self.p.view})
        else:
            self.stop_time_march(None)
            run_init(self.p)

    def update_image(self):
        # Check for updates from the queue
        while not self.queue.empty():
            try:
                # Retrieve the image buffer from the queue
                png_buffer = self.queue.get()
                png_buffer.seek(0)  # Ensure the buffer is at the start

                core_img = CoreImage(png_buffer, ext="png")
                core_img.texture.min_filter = "nearest"
                core_img.texture.mag_filter = "nearest"
                self.img.texture = core_img.texture
                self.img.canvas.ask_update()  # Force the image widget to redraw
            except Exception as e:
                print(f"Error updating image: {e}")

    def init_time_march(self, instance):
        self.process = multiprocessing.Process(target=run_init, args=(self.p,))
        self.process.start()

    def start_time_march(self, instance):
        if self.process is not None:
            self.stop_time_march(instance)
        self.process = multiprocessing.Process(target=run_time_march, args=(self.p,))
        self.process.start()

    def stop_time_march(self, instance):
        if self.process is not None:
            self.stop_event.set()
            self.process.join()
            self.stop_event.clear()
            print("Process terminated")

    def on_stop(self):
        self.stop_time_march(None)
        # Kill the process when the app closes
        os.kill(self.process.pid, signal.SIGTERM)

    def save_state(self):
        self.queue2.put("Save state")

    def load_state(self):
        self.queue2.put("Load state")

    def load_charge_discharge(self):
        home_directory = os.path.expanduser("~")

        # Create a file chooser popup for selecting CSV files, setting the default path to the home directory
        file_chooser = FileChooserListView(filters=["*.csv"], path=home_directory, size_hint=(0.9, 0.9))

        popup = Popup(title="Select CSV File", content=file_chooser, size_hint=(0.9, 0.9))

        file_chooser.bind(on_submit=lambda chooser, selection, touch: self.on_file_select(selection, popup))

        popup.open()

    def on_file_select(self, selection, popup):
        # Close the file chooser popup after selection
        popup.dismiss()

        if selection:
            file_path = selection[0]
            try:
                with open(file_path, "r") as csv_file:
                    reader = csv.reader(csv_file)
                    data = list(reader)  # Read the CSV contents as a list of rows
                    self.queue2.put(data)  # Send the CSV data to queue2
                    # print(f"CSV data loaded and sent to queue2: {data}")
            except Exception as e:
                print(f"Error loading CSV file: {e}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    multiprocessing.freeze_support()
    if len(sys.argv) < 2:
        filename = params.resource_path("json/gui.json5")
    else:
        filename = sys.argv[1]
    with open(filename, "r") as f:
        data, p = params.load_file(f)
    p.concurrent_index = 0

    VoidMigrationApp(data, p).run()
