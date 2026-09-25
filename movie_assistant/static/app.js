const userSelect = document.querySelector('#user-select');
const chatLog = document.querySelector('#chat-log');
const form = document.querySelector('#chat-form');
const input = document.querySelector('#message-input');
const sendButton = document.querySelector('#send-button');
const typing = document.querySelector('#typing');
const toast = document.querySelector('#toast');
const themeToggle = document.querySelector('#theme-toggle');
let toastTimer;
let activeUserId = Number(userSelect.value);

function applyTheme(theme, persist = true) {
  document.documentElement.dataset.theme = theme;
  const isDark = theme === 'dark';
  document.querySelector('#theme-icon').textContent = isDark ? '☼' : '◐';
  document.querySelector('#theme-label').textContent = isDark ? 'Giao diện sáng' : 'Giao diện tối';
  themeToggle.setAttribute('aria-label', isDark ? 'Chuyển sang giao diện sáng' : 'Chuyển sang giao diện tối');
  document.querySelector('meta[name="theme-color"]').content = isDark ? '#131914' : '#f5f3ef';
  if (persist) {
    try { localStorage.setItem('movie-assistant-theme', theme); } catch { /* Keep the theme for this page view. */ }
  }
}

let savedTheme = null;
try { savedTheme = localStorage.getItem('movie-assistant-theme'); } catch { /* Storage may be disabled. */ }
const initialTheme = savedTheme === 'dark' || savedTheme === 'light'
  ? savedTheme
  : (window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
applyTheme(initialTheme, false);
themeToggle.addEventListener('click', () => {
  applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');
});

function showToast(message) {
  toast.textContent = message;
  toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove('show'), 2600);
}

function addMessage(role, text, error = false) {
  const wrapper = document.createElement('div');
  wrapper.className = `message ${role}`;
  const label = document.createElement('span');
  label.className = 'message-label';
  label.textContent = role === 'user' ? 'BẠN' : 'CUỐN PHIM';
  const bubble = document.createElement('div');
  bubble.className = `message-bubble${error ? ' error' : ''}`;
  bubble.textContent = text;
  wrapper.append(label, bubble);
  chatLog.append(wrapper);
  return wrapper;
}

function addWelcome(userId) {
  const greeting = document.createElement('div');
  greeting.className = 'welcome-message';
  const kicker = document.createElement('span');
  kicker.className = 'welcome-kicker';
  kicker.textContent = `XIN CHÀO, USER ${userId}`;
  const copy = document.createElement('p');
  copy.textContent = 'Mình đã xem qua gu phim của bạn. Hôm nay ta khám phá gì nhỉ?';
  greeting.append(kicker, copy);
  chatLog.append(greeting);
}

function addRecommendations(items) {
  if (!items?.length) return;
  const list = document.createElement('div');
  list.className = 'movie-list';
  list.setAttribute('aria-label', 'Các phim được đề xuất');
  items.forEach((movie, index) => {
    const card = document.createElement('article');
    card.className = 'movie-card';
    const top = document.createElement('div');
    top.className = 'movie-card-top';
    const number = document.createElement('span');
    number.className = 'movie-number';
    number.textContent = String(index + 1).padStart(2, '0');
    const title = document.createElement('h2');
    title.className = 'movie-title';
    title.textContent = movie.title;
    if (movie.year) {
      const year = document.createElement('span');
      year.className = 'movie-year';
      year.textContent = ` (${movie.year})`;
      title.append(year);
    }
    const score = document.createElement('span');
    score.className = 'match-score';
    score.textContent = `Hạng ${index + 1}`;
    top.append(number, title, score);
    const genres = document.createElement('div');
    genres.className = 'movie-genres';
    (movie.genres || []).forEach((genre) => {
      const chip = document.createElement('span');
      chip.className = 'movie-tag';
      chip.textContent = genre;
      genres.append(chip);
    });
    const evidence = document.createElement('p');
    evidence.className = 'movie-evidence';
    evidence.textContent = movie.explanation;
    card.append(top, genres, evidence);
    list.append(card);
  });
  chatLog.append(list);
  const note = document.createElement('span');
  note.className = 'recommendation-note';
  note.textContent = 'Thứ tự xếp hạng dựa trên dữ liệu, không phải dự đoán số sao bạn sẽ chấm.';
  chatLog.append(note);
}

function scrollChat() {
  chatLog.scrollTop = chatLog.scrollHeight;
}

async function loadProfile(userId) {
  try {
    const response = await fetch(`/api/v1/users/${encodeURIComponent(userId)}`);
    const profile = await response.json();
    if (!response.ok) throw new Error(profile.error?.message || 'Không tải được hồ sơ.');
    if (!userSelect.dataset.loaded) {
      const listResponse = await fetch('/api/v1/users');
      const list = await listResponse.json();
      if (!listResponse.ok) throw new Error(list.error?.message || 'Không tải được danh sách hồ sơ.');
      const selected = String(userId);
      const preferred = new Set([1, 15, 30]);
      const ids = list.user_ids.slice().sort((a, b) => {
        const aPriority = preferred.has(a) ? [1, 15, 30].indexOf(a) : 3;
        const bPriority = preferred.has(b) ? [1, 15, 30].indexOf(b) : 3;
        return aPriority - bPriority || a - b;
      });
      userSelect.replaceChildren(...ids.map((id) => {
        const option = document.createElement('option');
        option.value = id;
        option.textContent = `User ${id}${preferred.has(id) ? ' · mẫu' : ''}`;
        return option;
      }));
      userSelect.value = selected;
      userSelect.dataset.loaded = 'true';
    }
    document.querySelector('#profile-monogram').textContent = userId;
    document.querySelector('#profile-title').textContent = `Người xem #${userId}`;
    document.querySelector('#profile-count').textContent = `${profile.rating_count} lượt đánh giá`;
    document.querySelector('#profile-average').innerHTML = `${profile.average_rating.toFixed(2)} <small>/ 5</small>`;
    document.querySelector('#welcome-user').textContent = userId;
    const genreBox = document.querySelector('#profile-genres');
    genreBox.replaceChildren();
    const genres = profile.favorite_genres.length ? profile.favorite_genres : ['Chưa đủ dữ liệu'];
    genres.forEach((genre) => {
      const chip = document.createElement('span');
      chip.className = `genre-chip${genre === 'Chưa đủ dữ liệu' ? ' muted-chip' : ''}`;
      chip.textContent = genre;
      genreBox.append(chip);
    });
  } catch (error) {
    showToast(error.message);
  }
}

async function resetContext(userId) {
  try {
    await fetch(`/api/v1/users/${encodeURIComponent(userId)}/context`, { method: 'DELETE' });
  } catch {
    // Starting a fresh visual chat still works if context cleanup is unavailable.
  }
}

async function sendMessage(message) {
  const cleaned = message.trim();
  if (!cleaned || sendButton.disabled) return;
  document.querySelector('.welcome-message')?.remove();
  addMessage('user', cleaned);
  input.value = '';
  input.style.height = 'auto';
  sendButton.disabled = true;
  typing.hidden = false;
  scrollChat();
  try {
    const response = await fetch('/api/v1/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: Number(userSelect.value), message: cleaned }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error?.message || 'Chưa gửi được câu hỏi.');
    if (result.intent === 'recommend' && result.recommendations?.length) {
      addMessage('assistant', result.headline);
      addRecommendations(result.recommendations);
    } else {
      addMessage('assistant', result.answer);
    }
  } catch (error) {
    addMessage('assistant', `Không kết nối được với trợ lý. ${error.message}`, true);
  } finally {
    typing.hidden = true;
    sendButton.disabled = false;
    input.focus();
    scrollChat();
  }
}

form.addEventListener('submit', (event) => {
  event.preventDefault();
  sendMessage(input.value);
});

input.addEventListener('input', () => {
  input.style.height = 'auto';
  input.style.height = `${Math.min(input.scrollHeight, 120)}px`;
});

input.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

document.querySelectorAll('[data-prompt]').forEach((button) => {
  button.addEventListener('click', () => sendMessage(button.dataset.prompt));
});

userSelect.addEventListener('change', async () => {
  sendButton.disabled = true;
  await resetContext(activeUserId);
  activeUserId = Number(userSelect.value);
  chatLog.replaceChildren();
  addWelcome(userSelect.value);
  await loadProfile(userSelect.value);
  sendButton.disabled = false;
});

document.querySelector('#clear-chat').addEventListener('click', async () => {
  await resetContext(activeUserId);
  chatLog.replaceChildren();
  addWelcome(userSelect.value);
});

loadProfile(userSelect.value);
