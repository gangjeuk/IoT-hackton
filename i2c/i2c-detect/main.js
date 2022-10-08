// i2cDetect的なものを実装してみます
// https://kernel.googlesource.com/pub/scm/utils/i2c-tools/i2c-tools/+/v3.1.2/tools/i2cdetect.c
// によると、 readmodeの場合はi2c_smbus_read_byteしてるだけなんですね・・

var i2cPort;

main();

async function main() {
  makeTable();
  var i2cAccess = await navigator.requestI2CAccess();
  i2cPort = i2cAccess.ports.get(1);
  readData();
}

async function readData() {
  for (var i = 0; i < 128; i++) {
    document.getElementById("ADDR" + i).innerText = "";
  }
  for (var slaveAddress = 0; slaveAddress < 128; slaveAddress++) {
    var i2cSlave = await i2cPort.open(slaveAddress);
    try {
      var ret = await i2cSlave.readBytes(32);
      // エラーが起きるところはつながってない
      console.log("addr:", slaveAddress, "  ans:", ret);
      document.getElementById(
        "ADDR" + slaveAddress
      ).innerText = slaveAddress.toString(16);
    } catch (e) {
      document.getElementById("ADDR" + slaveAddress).innerText = "--";
      //			console.log("addr:",slaveAddress,"  ans: ERROR");
    }
    await sleep(10);
  }
}

function makeTable() {
  var tbl = document.createElement("table");
  var addr = 0;
  for (var i = 0; i < 9; i++) {
    var tr = document.createElement("tr");

    for (var j = 0; j < 17; j++) {
      var td = document.createElement("td");
      if (i == 0) {
        if (j == 0) {
        } else {
          td.innerText = (j - 1).toString(16);
        }
      } else {
        if (j == 0) {
          td.innerText = ((i - 1) * 16).toString(16);
        } else {
          td.id = "ADDR" + addr;
          ++addr;
        }
      }
      tr.appendChild(td);
    }
    tbl.appendChild(tr);
  }
  var detect = document.getElementById("detect");
  detect.appendChild(tbl);
}
