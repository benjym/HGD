pyinstaller \
    --onefile \
    --windowed \
    void_migration/gui.py \
    --paths ./void_migration/ \
    --collect-all void_migration \
    --hidden-import kivymd.icon_definitions \
    --name VoidMigration \
    --add-data "json/*:json" \
    --icon assets/favicon.png