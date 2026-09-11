(function () {
  "use strict";

  const PIECES = {
    K: "♔", Q: "♕", R: "♖", B: "♗", N: "♘", P: "♙",
    k: "♚", q: "♛", r: "♜", b: "♝", n: "♞", p: "♟"
  };
  const FILES = "abcdefgh";
  const boardElement = document.getElementById("board");
  const turnElement = document.getElementById("turn");
  const statusPill = document.getElementById("status-pill");
  const statusMessage = document.getElementById("status-message");
  const moveCount = document.getElementById("move-count");
  const moveCountLabel = document.getElementById("move-count-label");
  const outcomeElement = document.getElementById("outcome");
  const boundElement = document.getElementById("bound");
  const fenElement = document.getElementById("fen");
  const thoughtsElement = document.getElementById("thoughts");
  const thoughtCountElement = document.getElementById("thought-count");
  const movesElement = document.getElementById("moves");
  const form = document.getElementById("game-form");
  const startButton = document.getElementById("start-button");
  const errorElement = document.getElementById("error");
  const orientationElement = document.getElementById("orientation");
  const policyBadge = document.getElementById("policy-badge");
  const surrogateNote = document.getElementById("surrogate-note");
  const flyColorInput = document.getElementById("fly-color");
  const depthInput = document.getElementById("depth");
  const maxPliesInput = document.getElementById("max-plies");

  function parseFen(fen) {
    const placement = (fen || "").split(" ")[0] || "8/8/8/8/8/8/8/8";
    const rows = placement.split("/");
    const pieces = {};
    rows.forEach(function (row, rowIndex) {
      let file = 0;
      Array.from(row).forEach(function (token) {
        if (/^[1-8]$/.test(token)) {
          file += Number(token);
        } else if (file < 8) {
          pieces[FILES[file] + (8 - rowIndex)] = token;
          file += 1;
        }
      });
    });
    return pieces;
  }

  function lastMoveSquares(state) {
    const recent = state.recent_moves || [];
    const last = recent[recent.length - 1];
    if (!last || !last.uci || last.uci.length < 4) return [];
    return [last.uci.slice(0, 2), last.uci.slice(2, 4)];
  }

  function renderBoard(state) {
    const pieces = parseFen(state.fen);
    const highlighted = lastMoveSquares(state);
    boardElement.replaceChildren();
    const flyColor = state.fly && state.fly.color ? state.fly.color : "white";

    const rankStep = flyColor === "black" ? 1 : -1;
    const fileStep = flyColor === "black" ? -1 : 1;
    const firstRank = flyColor === "black" ? 0 : 7;
    const firstFile = flyColor === "black" ? 7 : 0;
    orientationElement.textContent = flyColor === "black" ? "Black perspective" : "White perspective";

    for (let rankIndex = 0; rankIndex < 8; rankIndex += 1) {
      const rank = firstRank + rankIndex * rankStep;
      for (let fileIndex = 0; fileIndex < 8; fileIndex += 1) {
        const file = firstFile + fileIndex * fileStep;
        const square = FILES[file] + (rank + 1);
        const cell = document.createElement("div");
        cell.className = "square " + ((file + rank) % 2 === 0 ? "light" : "dark");
        cell.setAttribute("role", "gridcell");
        const piece = pieces[square];
        const side = piece && piece === piece.toUpperCase() ? "white" : "black";
        cell.setAttribute("aria-label", square + (piece ? " " + side + " piece" : " empty"));
        if (highlighted.indexOf(square) !== -1) cell.classList.add("last-move");
        if (piece && PIECES[piece]) {
          const glyph = document.createElement("span");
          glyph.className = "piece " + side;
          glyph.textContent = PIECES[piece];
          glyph.setAttribute("aria-hidden", "true");
          cell.appendChild(glyph);
        }
        boardElement.appendChild(cell);
      }
    }
  }

  function renderMoves(state) {
    const recent = state.recent_moves || [];
    movesElement.replaceChildren();
    if (!recent.length) {
      const empty = document.createElement("li");
      empty.className = "empty-state";
      empty.textContent = "No moves yet. Start an experiment.";
      movesElement.appendChild(empty);
      return;
    }
    recent.forEach(function (move) {
      const item = document.createElement("li");
      item.className = move.actor === "fly" ? "fly" : "stockfish";
      const actor = document.createElement("span");
      actor.className = "move-actor";
      actor.textContent = move.actor === "fly" ? "fly policy" : "Stockfish";
      const san = document.createElement("strong");
      san.textContent = move.san || move.uci || "?";
      const uci = document.createElement("span");
      uci.className = "move-uci";
      uci.textContent = move.uci || "";
      item.append(actor, san, uci);
      movesElement.appendChild(item);
    });
  }

  function activityLabel(activity) {
    const labels = {
      active_nodes: "active",
      node_count: "nodes",
      output_spikes: "output spikes",
      mean_abs: "mean",
      peak: "peak"
    };
    return Object.keys(activity || {}).map(function (key) {
      const value = Number(activity[key]);
      if (!Number.isFinite(value)) return null;
      const formatted = Number.isInteger(value) ? String(value) : value.toFixed(3);
      return (labels[key] || key.replace(/_/g, " ")) + " " + formatted;
    }).filter(Boolean).join(" · ");
  }

  function renderThoughts(state) {
    const trace = state.decision_trace || [];
    thoughtsElement.replaceChildren();
    thoughtCountElement.textContent = String(trace.length) +
      (trace.length === 1 ? " decision" : " decisions");
    if (!trace.length) {
      const empty = document.createElement("p");
      empty.className = "empty-state";
      empty.textContent = "No fly decisions yet. Start an experiment.";
      thoughtsElement.appendChild(empty);
      return;
    }

    trace.slice().reverse().forEach(function (decision) {
      const block = document.createElement("article");
      block.className = "thought-block";

      const heading = document.createElement("div");
      heading.className = "thought-heading";
      const turn = document.createElement("span");
      turn.textContent = "PLY " + String(decision.ply || "?");
      const selected = document.createElement("strong");
      const selectedCandidate = (decision.candidates || []).find(function (candidate) {
        return candidate.uci === decision.selected_uci;
      });
      selected.textContent = "selected " +
        (selectedCandidate && selectedCandidate.san || decision.selected_uci || "?");
      heading.append(turn, selected);
      block.appendChild(heading);

      const activity = activityLabel(decision.activity);
      if (activity) {
        const activityLine = document.createElement("p");
        activityLine.className = "thought-activity";
        activityLine.textContent = activity;
        block.appendChild(activityLine);
      }

      const candidates = decision.candidates || [];
      const scores = candidates.map(function (candidate) { return Number(candidate.score); });
      const finiteScores = scores.filter(Number.isFinite);
      const minimum = finiteScores.length ? Math.min.apply(null, finiteScores) : 0;
      const maximum = finiteScores.length ? Math.max.apply(null, finiteScores) : 1;
      const range = maximum - minimum;
      const candidateList = document.createElement("div");
      candidateList.className = "candidate-list";
      candidates.forEach(function (candidate, index) {
        const row = document.createElement("div");
        row.className = "candidate-row" +
          (candidate.uci === decision.selected_uci ? " selected" : "");
        const label = document.createElement("div");
        label.className = "candidate-label";
        const move = document.createElement("strong");
        move.textContent = candidate.san || candidate.uci || "?";
        const uci = document.createElement("span");
        uci.textContent = candidate.uci || "";
        label.append(move, uci);
        const score = document.createElement("span");
        score.className = "candidate-score";
        const numericScore = Number(candidate.score);
        score.textContent = Number.isFinite(numericScore) ? numericScore.toFixed(3) : "—";
        const bar = document.createElement("span");
        const normalized = range > 0 && Number.isFinite(numericScore) ?
          (numericScore - minimum) / range : 0.65;
        const level = Math.max(1, Math.min(10, Math.round(normalized * 9) + 1));
        bar.className = "candidate-bar level-" + String(level);
        row.append(label, score, bar);
        candidateList.appendChild(row);
      });
      block.appendChild(candidateList);
      thoughtsElement.appendChild(block);
    });
  }

  function formatStatus(status) {
    return (status || "ready").replace(/_/g, " ").toUpperCase();
  }

  function renderState(state) {
    if (!state || !state.fen) return;
    renderBoard(state);
    renderMoves(state);
    renderThoughts(state);
    const turn = (state.fen.split(" ")[1] || "w") === "w" ? "White" : "Black";
    turnElement.textContent = turn + " to move";
    statusPill.textContent = formatStatus(state.status);
    statusPill.dataset.status = state.status || "ready";
    statusMessage.textContent = state.message || "";
    const count = Number(state.move_count || 0);
    moveCount.textContent = String(count);
    moveCountLabel.textContent = String(count) + (count === 1 ? " ply" : " plies");
    outcomeElement.textContent = state.outcome && state.outcome !== "*" ? state.outcome : "—";
    boundElement.textContent = String(state.bounds && state.bounds.max_plies ? state.bounds.max_plies : "—");
    fenElement.textContent = state.fen;
    const fly = state.fly || {};
    if (fly.color === "white" || fly.color === "black") flyColorInput.value = fly.color;
    if (state.stockfish && Number.isInteger(state.stockfish.depth)) {
      depthInput.value = String(state.stockfish.depth);
    }
    if (state.bounds && Number.isInteger(state.bounds.max_plies)) {
      maxPliesInput.value = String(state.bounds.max_plies);
    }
    policyBadge.querySelector("span:last-child").textContent = "SIMULATION · " +
      String(fly.policy || "UNKNOWN POLICY").toUpperCase();
    surrogateNote.hidden = fly.is_surrogate !== true;
  }

  function showError(message) {
    errorElement.textContent = message || "The request failed.";
    errorElement.hidden = false;
  }

  function clearError() {
    errorElement.textContent = "";
    errorElement.hidden = true;
  }

  async function readJson(response) {
    let payload;
    try {
      payload = await response.json();
    } catch (_error) {
      throw new Error("The local server returned an unreadable response.");
    }
    if (!response.ok) {
      if (payload.state) renderState(payload.state);
      throw new Error(payload.error || "The local server rejected the request.");
    }
    return payload;
  }

  async function loadState() {
    try {
      const response = await fetch("/api/state", { headers: { "Accept": "application/json" } });
      renderState(await readJson(response));
    } catch (error) {
      showError(error.message);
    }
  }

  form.addEventListener("submit", async function (event) {
    event.preventDefault();
    clearError();
    startButton.disabled = true;
    startButton.querySelector("span:last-child").textContent = "Running bounded game…";
    try {
      const body = {
        fly_color: flyColorInput.value,
        depth: Number(depthInput.value),
        max_plies: Number(maxPliesInput.value)
      };
      const response = await fetch("/api/game", {
        method: "POST",
        headers: { "Accept": "application/json", "Content-Type": "application/json" },
        body: JSON.stringify(body)
      });
      renderState(await readJson(response));
    } catch (error) {
      showError(error.message);
    } finally {
      startButton.disabled = false;
      startButton.querySelector("span:last-child").textContent = "Start bounded game";
    }
  });

  loadState();
}());
