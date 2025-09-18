let pedido = [];

// Coordenadas da lanchonete (SJC)
const LANCHONETE_LAT = -23.1896;
const LANCHONETE_LNG = -45.8841;

// Taxas de entrega
const taxasEntrega = {
  "Alto da Ponte": 4,
  "Altos da Vila Paiva": 6,
  "Altos de Santana": 3,
  "Altos do Caete": 6,
  "Vila Unidos": 3,
  "Caete": 6,
  "Centro": 6,
  "Jardim Guimarães": 4,
  "JD Minas Gerais": 4,
  "Monte Castelo": 7,
  "Petybon": 7,
  "Santana": 5,
  "Telespark": 2,
  "Vila Cristina": 6,
  "Vila Dirce": 3,
  "Vila Jaci": 2,
  "Vila Machado": 5,
  "Vila Paiva": 5,
  "Vila São Geraldo": 5,
  "Vila Sinha": 2
};

// Adicionar item
function adicionarItem(nome, preco) {
  pedido.push({ nome, preco });
  showToast(`${nome} adicionado ao pedido!`);
  atualizarResumo();
}

// Remover item
function removerItem(index) {
  if (index >= 0 && index < pedido.length) {
    pedido.splice(index, 1);
    atualizarResumo();
  }
}

// Função Haversine para calcular distância
function calcularDistancia(lat1, lon1, lat2, lon2) {
  const R = 6371; // km
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLon = (lon2 - lon1) * Math.PI / 180;
  const a = Math.sin(dLat / 2) ** 2 +
            Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
            Math.sin(dLon / 2) ** 2;
  return R * (2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a)));
}

// Pegar taxa de entrega
async function getTaxaEntrega(bairro) {
  if (bairro && taxasEntrega[bairro] !== undefined) {
    return taxasEntrega[bairro];
  }

  if (bairro === "rural") {
    return new Promise((resolve) => {
      if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(pos => {
          const distancia = calcularDistancia(
            LANCHONETE_LAT,
            LANCHONETE_LNG,
            pos.coords.latitude,
            pos.coords.longitude
          );
          const taxa = 3 + Math.ceil(distancia);
          resolve(taxa);
        }, () => {
          alert("Não foi possível pegar sua localização. Será aplicada taxa fixa R$3.");
          resolve(3);
        });
      } else {
        alert("Seu navegador não suporta geolocalização. Será aplicada taxa fixa R$3.");
        resolve(3);
      }
    });
  }

  return 0;
}

// Atualizar resumo
async function atualizarResumo() {
  const lista = document.getElementById("lista-pedido");
  const totalEl = document.getElementById("total");
  lista.innerHTML = "";

  let total = pedido.reduce((soma, item) => soma + item.preco, 0);

  pedido.forEach((item, index) => {
    const li = document.createElement("li");
    li.innerHTML = `
      ${item.nome} - R$ ${item.preco.toFixed(2)}
      <button class="btn-remover" data-index="${index}" aria-label="Remover ${item.nome}">Remover</button>
    `;
    lista.appendChild(li);
  });

  // pegar bairro
  const bairro = document.getElementById("bairro")?.value;
  let taxa = bairro ? await getTaxaEntrega(bairro) : 0;

  if (bairro) {
    const taxaEl = document.createElement("li");
    taxaEl.innerHTML = `🚚 Taxa de Entrega (${bairro}) - R$ ${taxa.toFixed(2)}`;
    lista.appendChild(taxaEl);
  }

  total += taxa;
  totalEl.innerText = `Total: R$ ${total.toFixed(2)}`;

  lista.querySelectorAll(".btn-remover").forEach(btn => {
    btn.onclick = () => {
      const idx = Number(btn.getAttribute("data-index"));
      removerItem(idx);
    };
  });
}

// Finalizar pedido
function finalizarPedido() {
  if (pedido.length === 0) {
    showToast("Adicione itens ao pedido antes de finalizar.");
    return;
  }

  const modal = document.getElementById("modal-confirmacao");
  modal.classList.add("show");

  const btnConfirmar = document.getElementById("btn-confirmar");
  const btnCancelar = document.getElementById("btn-cancelar");

  btnConfirmar.replaceWith(btnConfirmar.cloneNode(true));
  btnCancelar.replaceWith(btnCancelar.cloneNode(true));

  const novoConfirmar = document.getElementById("btn-confirmar");
  const novoCancelar = document.getElementById("btn-cancelar");

  novoCancelar.onclick = () => modal.classList.remove("show");
  novoConfirmar.onclick = () => {
    enviarPedidoWhatsApp();
    modal.classList.remove("show");
  };
}

// Enviar pedido WhatsApp
async function enviarPedidoWhatsApp() {
  const tipo = document.getElementById("tipo").value;
  const endereco = document.getElementById("endereco")?.value.trim() || "";
  const pagamento = document.getElementById("pagamento")?.value || "";
  const bairro = document.getElementById("bairro")?.value;

  let mensagem = "*Pedido Fulanos Hamburgueria*%0A";

  pedido.forEach(item => {
    mensagem += `• ${item.nome} - R$ ${item.preco.toFixed(2)}%0A`;
  });

  let total = pedido.reduce((soma, item) => soma + item.preco, 0);

  // Adiciona taxa de entrega
  let taxa = bairro ? await getTaxaEntrega(bairro) : 0;
  if (taxa > 0) {
    mensagem += `🚚 Taxa de entrega (${bairro}) - R$ ${taxa.toFixed(2)}%0A`;
    total += taxa;
  }

  mensagem += `%0ATotal: R$ ${total.toFixed(2)}%0A`;

  if (tipo === "entrega") {
    if (!endereco) {
      alert("Por favor, preencha o endereço para entrega.");
      return;
    }
    mensagem += `Tipo de pedido: *ENTREGA*%0A📍 Endereço: ${encodeURIComponent(endereco)}%0A`;
  } else {
    mensagem += "Tipo de pedido: *RETIRADA*%0A";
  }

  let pagamentoTexto = pagamento === "dinheiro" ? "Dinheiro" :
                        pagamento === "cartao" ? "Cartão" :
                        pagamento === "pix" ? "PIX" : "Não especificado";
  mensagem += `Forma de pagamento: *${pagamentoTexto}*%0A`;

  const numero = "5512997481692";
  const link = `https://wa.me/${numero}?text=${mensagem}`;

  showToast("Redirecionando para o WhatsApp...");
  window.location.href = link;

  limparPedido();
}

// Limpar pedido
function limparPedido() {
  pedido = [];
  atualizarResumo();
}

// Toast
function showToast(msg) {
  const toast = document.getElementById("toast");
  toast.textContent = msg;
  toast.classList.add("show");
  if (toast.hideTimeout) clearTimeout(toast.hideTimeout);
  toast.hideTimeout = setTimeout(() => toast.classList.remove("show"), 2500);
}

// Mostrar campo de endereço
document.getElementById("tipo").addEventListener("change", function () {
  const campoEndereco = document.getElementById("campo-endereco");
  campoEndereco.style.display = this.value === "entrega" ? "block" : "none";
});
