// Heart Disease Predictor - Custom JavaScript

document.addEventListener('DOMContentLoaded', function() {

    // ---------- Password visibility toggle (login & register) ----------
    const togglePassword = document.getElementById('togglePassword');
    if (togglePassword) {
        togglePassword.addEventListener('click', function() {
            const passwordInput = this.closest('.input-group').querySelector('input[type="password"]');
            if (passwordInput) {
                const type = passwordInput.getAttribute('type') === 'password' ? 'text' : 'password';
                passwordInput.setAttribute('type', type);
                this.querySelector('i').classList.toggle('fa-eye');
                this.querySelector('i').classList.toggle('fa-eye-slash');
            }
        });
    }

    // ---------- Confirm password validation on register ----------
    const registerForm = document.querySelector('form[action*="register"]');
    if (registerForm) {
        registerForm.addEventListener('submit', function(e) {
            const password = document.getElementById('password');
            const confirm = document.getElementById('confirm_password');
            if (password && confirm && password.value !== confirm.value) {
                e.preventDefault();
                alert('Passwords do not match. Please re-enter.');
                confirm.focus();
            }
        });
    }

    // ---------- Auto-dismiss alerts after 5 seconds ----------
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(alert => {
        setTimeout(() => {
            const closeBtn = alert.querySelector('.btn-close');
            if (closeBtn) closeBtn.click();
        }, 5000);
    });

    // ---------- Prediction form: confirm before submit (optional) ----------
    const predictForm = document.querySelector('form[action*="predict"]');
    if (predictForm) {
        predictForm.addEventListener('submit', function(e) {
            // You can add client-side validation here if needed
            // For now, just show a loading spinner or something
            const btn = this.querySelector('button[type="submit"]');
            if (btn) {
                btn.disabled = true;
                btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Predicting...';
            }
        });
    }

    // ---------- Delete confirmation (history page) ----------
    const deleteLinks = document.querySelectorAll('a[href*="delete_history"]');
    deleteLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            if (!confirm('Are you sure you want to delete this prediction record?')) {
                e.preventDefault();
            }
        });
    });

    // ---------- Tooltip initialization (if needed) ----------
    // var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    // var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
    //     return new bootstrap.Tooltip(tooltipTriggerEl);
    // });

    console.log('Heart Disease Predictor loaded successfully.');
});