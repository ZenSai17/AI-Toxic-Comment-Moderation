function warnUser(userId) {
    fetch('/warn-user', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: userId })
    })
    .then(response => response.json())
    .then(data => alert(data.status))
    .catch(() => alert('Error warning user'));
}

function banUser(userId) {
    fetch('/ban-user', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: userId })
    })
    .then(response => response.json())
    .then(data => alert(data.status))
    .catch(() => alert('Error banning user'));
}


function updateMessageCount(messages) {
    const toxicMessages = messages.filter(msg => msg.prediction === 'Toxic Comment');
    const nonToxicMessages = messages.filter(msg => msg.prediction === 'Non-Toxic Comment');

    document.getElementById('toxicCount').textContent = `Toxic Messages: ${toxicMessages.length}`;
    document.getElementById('nonToxicCount').textContent = `Non-Toxic Messages: ${nonToxicMessages.length}`;
}


function updateMessageTable(messages) {
    const tableBody = document.querySelector('#messagesTable tbody');
    tableBody.innerHTML = '';

    messages.forEach(msg => {
        const row = document.createElement('tr');
        row.classList.add(msg.prediction.toLowerCase().replace(' ', '-'));

        row.innerHTML = `
            <td>${msg.text}</td>
            <td>${msg.user_id}</td>
            <td>${msg.prediction}</td>
            <td>
                <button class="btn btn-warning" onclick="warnUser('${msg.user_id}')">Warn</button>
                <button class="btn btn-ban" onclick="banUser('${msg.user_id}')">Ban</button>
            </td>
        `;
        tableBody.appendChild(row);
    });
}


function fetchLatestMessages() {
    document.getElementById('loadingMessage').style.display = 'block';
    fetch('/get-latest-flagged-messages')
        .then(response => response.json())
        .then(data => {
            updateMessageCount(data.messages);
            updateMessageTable(data.messages);
            document.getElementById('loadingMessage').style.display = 'none';
        })
        .catch(() => {
            alert('Error fetching messages');
            document.getElementById('loadingMessage').style.display = 'none';
        });
}


setInterval(fetchLatestMessages, 5000);
fetchLatestMessages(); 
