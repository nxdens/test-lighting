import json
import os
from matplotlib import container


class ConfigContainer(dict):
    """Class to load dict from json file

    Extends python dict to load from json files and adds usage of the `.`
    operator to call/set keys and values. When the dictonary is updated
    the json_file is updated to record the change. Print
    Some code referenced from
    [stackoverflow](https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary).

    Typical usage example:

    ```
    container = ConfigContainer(jsonfile)
    container.key
    container.key.subkey
    container.key = "something"
    del container.key
    container.update_json()
    ```

    """

    json_file: str = None
    """reference json file path
    """

    def __init__(self, json_filename: str = None, *args, **kwargs):
        super(ConfigContainer, self).__init__(*args, **kwargs)

        if json_filename is not None:
            self.set_json(json_filename)
            self.read_json()

        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    if isinstance(v, dict):
                        self[k] = ConfigContainer(None, v)
                    else:
                        self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                if isinstance(v, dict):
                    self[k] = ConfigContainer(None, v)
                else:
                    self[k] = v

    def set_json(self, filename: str) -> None:
        """Sets json filename for the container class

        Args:
            filename (str): path for the json file
        """
        self.json_file = filename

    def read_json(self) -> None:
        """Read in parameters from json file

        Read each level of dictionary as its own container recursively.
        """
        with open(self.json_file) as f:
            data = json.load(f)
            for k, v in data.items():
                if isinstance(v, dict):
                    self[k] = ConfigContainer(None, v)
                else:
                    self[k] = v

    def update_json(self) -> None:
        """Dumps parameters to json file
        """
        with open(self.json_file, "w") as f:
            dict_copy = self.copy()
            del dict_copy["json_file"]
            json.dump(dict_copy, f)

    def __getattr__(self, attr) -> None:
        """override dict '.' operator for access"""
        return self.get(attr)

    def __setattr__(self, key, value) -> None:
        """override dict '.' operator for setting keys"""
        self.__setitem__(key, value)

    def __setitem__(self, key, value) -> None:
        """allows use of '.' operator for setting key values"""
        if isinstance(value, dict):
            value = ConfigContainer(None, value)
        super(ConfigContainer, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(ConfigContainer, self).__delitem__(key)
        del self.__dict__[key]

    def __str__(self) -> str:
        string = ["{"]
        for k, v in self.items():
            if isinstance(v, dict):
                strv = "\n\t".join(str(v).splitlines())
                string.append(f"\t{k}: " + f"{strv}")
            elif isinstance(v, list):
                strv = "\n\t\t".join(str(v).split(", "))
                string.append(f"\t{k}: " + f"{strv}")
            else:
                string.append(f"\t{k}: {v},")
        string.append("}")
        return "\t\n".join(string)


if __name__ == "__main__":
    """run unit tests
    """
    # should test access, update, delete
    test_dict = {"a": 123, "b": {"c": 456, "d": 789}}

    def setup(test_dict: dict):
        with open("dummy_file.json", "w") as f:
            json.dump(test_dict, f)

    setup(test_dict)
    _ = ConfigContainer("dummy_file.json")
    print("test access")
    print(container.json_file == "dummy_file.json")
    print(container.a == 123)
    print("test subdict/container")
    print(container.b == {"c": 456, "d": 789})
    print(container.b.c == 456 and container.b.d == 789)
    print("test update and delete")
    container.b.c = 999
    print(container.b.c == 999)
    del container.b.d
    print(container.b == {"c": 999})
    container.b.e = 927
    print(container.b == {"c": 999, "e": 927})
    container.b.f = {"g": 111}
    print(container.b.f == {"g": 111})
    print(container.b.f.g == 111)
    print("test update file")
    container.update_json()
    with open("dummy_file.json", "r") as f:
        data = json.load(f)
        print(data == {"a": 123, "b": {"c": 999, "e": 927, "f": {"g": 111}}})
    os.remove("dummy_file.json")
