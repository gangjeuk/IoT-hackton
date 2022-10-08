// プログラムの本体となる関数です。await で扱えるよう全体を async 関数で宣言します。
async function main() {
  // 非同期関数は await を付けて呼び出します。
  const button = document.getElementById("button");
  const ledView = document.getElementById("ledView");
  const gpioAccess = await navigator.requestGPIOAccess();
  const ledPort = gpioAccess.ports.get(26); // LED の GPIO ポート番号
  await ledPort.export("out");

  let lit = false;
  button.onclick = async function () {
    lit = !lit;

    await ledPort.write(lit ? 1 : 0);
    const color = lit ? "red" : "black";
    ledView.style.backgroundColor = color;
  };
}

// await sleep(ms) と呼ぶと、指定 ms (ミリ秒) 待機
// 同じものが polyfill.js でも定義されているため省略可能
function sleep(ms) {
  return new Promise(function (resolve) {
    setTimeout(resolve, ms);
  });
}

// 宣言した関数を実行します。このプログラムのエントリーポイントです。
main();
